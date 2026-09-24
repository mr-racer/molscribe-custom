"""
Greedy decoding for TransformerDecoderAR with a fixed batch and a preallocated KV cache.

The reference implementation (GreedySearch + onmt attention caches) drops finished sequences from the batch,
grows the self-attention cache with torch.cat and synchronizes with the host at every step. Here the batch never
shrinks (finished rows keep running and are ignored), keys/values are written into a preallocated cache, and the host
only checks for completion every `check_every` steps. On CUDA the whole decoding step can be captured once into a
CUDA graph and replayed, which removes the per-kernel launch overhead of the ~6 small transformer layers.

Semantics are the same as GreedySearch(min_length=1) + the tokenizer output constraint: output mask first, then EOS
is forbidden at the first step, then topk(1).
"""
import math

import torch
import torch.nn.functional as F

from ..tokenizer import SOS_ID, EOS_ID


class StaticGreedyDecoder:

    def __init__(self, ar_decoder, max_length, check_every=16, use_cuda_graph=False):
        self.m = ar_decoder
        self.max_length = max_length
        self.check_every = check_every
        self.use_cuda_graph = use_cuda_graph
        layers = ar_decoder.decoder.transformer_layers
        self.layers = list(layers)
        self.num_layers = len(layers)
        self.heads = layers[0].self_attn.head_count
        self.dim_per_head = layers[0].self_attn.dim_per_head
        self.dim = self.heads * self.dim_per_head
        self._graphs = {}

    @staticmethod
    def supported(ar_decoder):
        return all(layer.self_attn.max_relative_positions == 0 for layer in ar_decoder.decoder.transformer_layers)

    # ---- building blocks -------------------------------------------------------------------------------------------

    def _shape(self, x):
        b = x.size(0)
        return x.view(b, -1, self.heads, self.dim_per_head).transpose(1, 2)

    def _unshape(self, x):
        b = x.size(0)
        return x.transpose(1, 2).contiguous().view(b, -1, self.dim)

    def _attend(self, attn_module, query, keys, values, mask=None):
        query = self._shape(attn_module.linear_query(query)) / math.sqrt(self.dim_per_head)
        scores = torch.matmul(query, keys.transpose(2, 3)).float()
        if mask is not None:
            scores = scores.masked_fill(mask, -1e18)
        attn = torch.softmax(scores, dim=-1).to(query.dtype)
        return attn_module.final_linear(self._unshape(torch.matmul(attn, values)))

    def _memory_kv(self, memory_bank):
        kv = []
        for layer in self.layers:
            ca = layer.context_attn
            kv.append((self._shape(ca.linear_keys(memory_bank)), self._shape(ca.linear_values(memory_bank))))
        return kv

    def _embed(self, tokens):
        emb = self.m.embeddings.word_lut(tokens.view(-1, 1))  # (B, 1, D)
        pe = self.m.embeddings.make_embedding.pe
        # same arithmetic as PositionalEncoding.forward with every sequence at pe[0] (see TransformerDecoderAR.decode)
        return emb * math.sqrt(pe.dim) + pe.pe[0:1]

    def _step(self, tokens, step, memory_kv, cache_k, cache_v, self_mask=None):
        """One decoding step. `step` is an int (dynamic cache slice) or a 1-element tensor (masked full cache)."""
        x = self._embed(tokens)
        for i, layer in enumerate(self.layers):
            sa = layer.self_attn
            xn = layer.layer_norm_1(x)
            k = self._shape(sa.linear_keys(xn))
            v = self._shape(sa.linear_values(xn))
            if isinstance(step, int):
                cache_k[i][:, :, step:step + 1] = k
                cache_v[i][:, :, step:step + 1] = v
                keys, values = cache_k[i][:, :, :step + 1], cache_v[i][:, :, :step + 1]
            else:
                cache_k[i].index_copy_(2, step, k)
                cache_v[i].index_copy_(2, step, v)
                keys, values = cache_k[i], cache_v[i]
            query = self._attend(sa, xn, keys, values, self_mask) + x
            mem_k, mem_v = memory_kv[i]
            mid = self._attend(layer.context_attn, layer.layer_norm_2(query), mem_k, mem_v)
            x = layer.feed_forward(mid + query)
        hidden = self.m.decoder.layer_norm(x)  # (B, 1, D)
        log_probs = F.log_softmax(self.m.output_layer(hidden).squeeze(1), dim=-1)
        return hidden, log_probs

    def _select(self, log_probs, tokens, first_step):
        if self.m.tokenizer.output_constraint:
            log_probs = log_probs.masked_fill(self.m._output_mask_table[tokens], -10000)
        if isinstance(first_step, bool):
            if first_step:
                log_probs[:, EOS_ID] = -1e20
        else:
            log_probs[:, EOS_ID] = torch.where(first_step, torch.full_like(log_probs[:, EOS_ID], -1e20),
                                               log_probs[:, EOS_ID])
        return log_probs.topk(1, dim=-1)

    def _alloc(self, batch_size, dtype, device):
        shape = (batch_size, self.heads, self.max_length, self.dim_per_head)
        cache_k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(self.num_layers)]
        cache_v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(self.num_layers)]
        return cache_k, cache_v

    # ---- eager path ------------------------------------------------------------------------------------------------

    def _decode_eager(self, memory_bank):
        b, device = memory_bank.size(0), memory_bank.device
        memory_kv = self._memory_kv(memory_bank)
        cache_k, cache_v = self._alloc(b, memory_kv[0][0].dtype, device)
        tokens = torch.full((b,), SOS_ID, dtype=torch.long, device=device)
        out_tokens = torch.zeros((b, self.max_length), dtype=torch.long, device=device)
        out_logp = torch.zeros((b, self.max_length), dtype=torch.float, device=device)
        out_hidden = torch.zeros((b, self.max_length, self.dim), dtype=memory_bank.dtype, device=device)
        lengths = torch.zeros((b,), dtype=torch.long, device=device)
        finished = torch.zeros((b,), dtype=torch.bool, device=device)
        steps = self.max_length
        for step in range(self.max_length):
            hidden, log_probs = self._step(tokens, step, memory_kv, cache_k, cache_v)
            scores, ids = self._select(log_probs, tokens, step == 0)
            ids = ids.view(-1)
            out_tokens[:, step] = ids
            out_logp[:, step] = scores.view(-1).float()
            out_hidden[:, step] = hidden[:, 0].to(out_hidden.dtype)
            newly = ids.eq(EOS_ID) & ~finished
            lengths = torch.where(newly, torch.full_like(lengths, step + 1), lengths)
            finished = finished | newly
            tokens = ids
            if (step + 1) % self.check_every == 0 and bool(finished.all()):
                steps = step + 1
                break
        return out_tokens, out_logp, out_hidden, lengths, finished, steps

    # ---- CUDA graph path -------------------------------------------------------------------------------------------

    def _build_graph(self, batch_size, dtype, device, memory_shape):
        g = {}
        g["memory_kv"] = [(torch.zeros((batch_size, self.heads, memory_shape[1], self.dim_per_head), dtype=dtype,
                                       device=device),
                           torch.zeros((batch_size, self.heads, memory_shape[1], self.dim_per_head), dtype=dtype,
                                       device=device)) for _ in range(self.num_layers)]
        g["cache_k"], g["cache_v"] = self._alloc(batch_size, dtype, device)
        g["tokens"] = torch.full((batch_size,), SOS_ID, dtype=torch.long, device=device)
        g["step"] = torch.zeros((1,), dtype=torch.long, device=device)
        g["positions"] = torch.arange(self.max_length, device=device)
        g["out_tokens"] = torch.zeros((batch_size, self.max_length), dtype=torch.long, device=device)
        g["out_logp"] = torch.zeros((batch_size, self.max_length), dtype=torch.float, device=device)
        g["out_hidden"] = torch.zeros((batch_size, self.max_length, self.dim), dtype=dtype, device=device)
        g["lengths"] = torch.zeros((batch_size,), dtype=torch.long, device=device)
        g["finished"] = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        def body():
            step = g["step"]
            self_mask = (g["positions"] > step).view(1, 1, 1, -1)
            hidden, log_probs = self._step(g["tokens"], step, g["memory_kv"], g["cache_k"], g["cache_v"], self_mask)
            scores, ids = self._select(log_probs, g["tokens"], step.eq(0))
            g["out_tokens"].index_copy_(1, step, ids)
            g["out_logp"].index_copy_(1, step, scores.float())
            g["out_hidden"].index_copy_(1, step, hidden.to(dtype))
            ids = ids.view(-1)
            newly = ids.eq(EOS_ID) & ~g["finished"]
            g["lengths"].copy_(torch.where(newly, step + 1, g["lengths"]))
            g["finished"].copy_(g["finished"] | newly)
            g["tokens"].copy_(ids)
            g["step"].add_(1)

        # warm up on a side stream (required before capture), then capture one step
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                body()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        g["graph"] = graph
        return g

    def _decode_graph(self, memory_bank):
        b, device = memory_bank.size(0), memory_bank.device
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else memory_bank.dtype
        key = (b, dtype, memory_bank.size(1))
        if key not in self._graphs:
            self._graphs[key] = self._build_graph(b, dtype, device, memory_bank.shape)
        g = self._graphs[key]
        for (dst_k, dst_v), (src_k, src_v) in zip(g["memory_kv"], self._memory_kv(memory_bank)):
            dst_k.copy_(src_k)
            dst_v.copy_(src_v)
        g["tokens"].fill_(SOS_ID)
        g["step"].zero_()
        g["lengths"].zero_()
        g["finished"].zero_()
        steps = self.max_length
        for step in range(self.max_length):
            g["graph"].replay()
            if (step + 1) % self.check_every == 0 and bool(g["finished"].all()):
                steps = step + 1
                break
        return g["out_tokens"], g["out_logp"], g["out_hidden"], g["lengths"], g["finished"], steps

    # ---- public ----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, memory_bank):
        """Returns (predictions, scores, token_scores, hidden) in the format of TransformerDecoderAR.decode."""
        use_graph = self.use_cuda_graph and memory_bank.is_cuda
        decode = self._decode_graph if use_graph else self._decode_eager
        out_tokens, out_logp, out_hidden, lengths, finished, steps = decode(memory_bank)
        # sequences that never produced EOS were stopped at max_length (GreedySearch.ensure_max_length)
        lengths = torch.where(finished, lengths, torch.full_like(lengths, steps)).tolist()
        tokens_cpu = out_tokens[:, :steps].cpu()
        logp_cpu = out_logp[:, :steps].cpu()
        predictions, scores, token_scores, hidden = [], [], [], []
        for i, length in enumerate(lengths):
            logp = logp_cpu[i, :length]
            predictions.append([tokens_cpu[i, :length]])
            scores.append([torch.exp(torch.mean(logp)).item()])
            token_scores.append([torch.exp(logp).tolist()])
            hidden.append([out_hidden[i, :length].clone() if use_graph else out_hidden[i, :length]])
        return predictions, scores, token_scores, hidden
