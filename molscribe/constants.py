from typing import List
import re

ORGANIC_SET = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']

# Variable-group placeholders as they appear in papers and patents: R with digit / prime / Greek / Latin suffixes,
# Ar', X', generic role labels (EWG, LG, Nu, PG ...). These become wildcard atoms, never expanded.
RGROUP_REGEX = re.compile(
    r"^(?:R(?:\d{0,2}|[a-d])|(?:Ar|Het|Alk|Hal|Cat|EWG|EDG|DG|LG|FG|PG|Nu|Nuc|El|E|M|L|X|Y|Z|Q|A|W|GROUP)\d{0,2})"
    r"(?:['′″]{0,2})(?:[α-ω]|\d{0,2})?(?:['′″]{0,2})[+-]?$|^\?$")


def is_rgroup(symbol):
    # a dictionary entry always wins over the placeholder pattern (Ac, Ad, ... are real groups)
    return symbol not in ABBREVIATIONS and (symbol in RGROUP_SYMBOLS or bool(RGROUP_REGEX.match(symbol)))


PLACEHOLDER_ATOMS = ["Lv", "Lu", "Nd", "Yb", "At", "Fm", "Er"]


class Substitution(object):
    '''Define common substitutions for chemical shorthand'''
    def __init__(self, abbrvs, smarts, smiles, probability, n_attach=1):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability
        # number of bonds the label makes in the drawing; the same label may have entries for several values
        # (CO: ligand with 1 bond, carbonyl with 2; THF: substituent with 1 bond, solvent molecule with 0)
        self.n_attach = n_attach


SUBSTITUTIONS: List[Substitution] = [
    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt', 'EtO2C'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),
    Substitution(['CO2Me', 'COOMe', 'MeO2C'], 'C(=O)[OH0;D2][CH3]', "[C](=O)OC", 0.5),

    Substitution(['OAc'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),

    Substitution(['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  # Benzoyl
    Substitution(['Bz', 'COPh', 'PhOC'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl

    Substitution(['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  # Benzyl
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl

    Substitution(['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[N]C(=O)OC(C)(C)C", 0.6),
    Substitution(['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),

    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)O[CH2][cH0]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;D3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),
    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['SEM'], '[CH2;D2]O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]OCC[Si](C)(C)C", 0.2),
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['TBZ'], 'C(=S)[cH0]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS'], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  # Tos

    # Alkyl chains
    Substitution(['OMe', 'MeO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[NH]C", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr', 'C3H7'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu', 'C4H9'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),
    Substitution(['C5H11', 'nPent', 'n-Pent'], '[CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCCC", 0.3),
    Substitution(['C6H13', 'nHex', 'n-Hex'], '[CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCCCC", 0.3),

    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    Substitution(['OCH3', 'H3CO'], '[OH0;D2][CH3]', "[O]C", 0.4),
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),
]


def _extra(abbrvs, smiles, n_attach=1):
    # Inference-only: expanded when predicted, never rendered into synthetic training images (no SMARTS).
    return Substitution(abbrvs, None, smiles, 0.0, n_attach)


EXTRA_SUBSTITUTIONS: List[Substitution] = [
    # Silyl / acetal / acyl protecting groups
    _extra(['TBDMS'], "[Si](C)(C)C(C)(C)C"),
    _extra(['OTBS', 'TBSO', 'OTBDMS', 'TBDMSO'], "[O][Si](C)(C)C(C)(C)C"),
    _extra(['TIPS'], "[Si](C(C)C)(C(C)C)C(C)C"),
    _extra(['OTIPS', 'TIPSO'], "[O][Si](C(C)C)(C(C)C)C(C)C"),
    _extra(['TBDPS'], "[Si](c1ccccc1)(c1ccccc1)C(C)(C)C"),
    _extra(['OTBDPS', 'TBDPSO'], "[O][Si](c1ccccc1)(c1ccccc1)C(C)(C)C"),
    _extra(['TES', 'SiEt3', 'Et3Si'], "[Si](CC)(CC)CC"),
    _extra(['OTES', 'TESO'], "[O][Si](CC)(CC)CC"),
    _extra(['SiMe3', 'Me3Si'], "[Si](C)(C)C"),
    _extra(['OTMS', 'TMSO'], "[O][Si](C)(C)C"),
    _extra(['SiPh3', 'Ph3Si'], "[Si](c1ccccc1)(c1ccccc1)c1ccccc1"),
    _extra(['MOM'], "[CH2]OC"),
    _extra(['OMOM', 'MOMO'], "[O]COC"),
    _extra(['MEM'], "[CH2]OCCOC"),
    _extra(['OMEM', 'MEMO'], "[O]COCCOC"),
    _extra(['OSEM', 'SEMO'], "[O]COCC[Si](C)(C)C"),
    _extra(['THP'], "[CH]1CCCCO1"),
    _extra(['OTHP', 'THPO'], "[O]C1CCCCO1"),
    _extra(['Piv', 'Pv'], "[C](=O)C(C)(C)C"),
    _extra(['OPiv', 'PivO', 'OPv', 'PvO'], "[O]C(=O)C(C)(C)C"),
    _extra(['Alloc'], "[C](=O)OCC=C"),
    _extra(['Troc'], "[C](=O)OCC(Cl)(Cl)Cl"),
    _extra(['Teoc'], "[C](=O)OCC[Si](C)(C)C"),
    _extra(['Tr', 'Trt'], "[C](c1ccccc1)(c1ccccc1)c1ccccc1"),
    _extra(['OTr', 'TrO', 'OTrt', 'TrtO'], "[O]C(c1ccccc1)(c1ccccc1)c1ccccc1"),
    _extra(['NHTr', 'TrHN', 'NHTrt', 'TrtHN'], "[NH]C(c1ccccc1)(c1ccccc1)c1ccccc1"),
    _extra(['OPMB', 'PMBO'], "[O]Cc1ccc(OC)cc1"),
    _extra(['PMP'], "[c]1ccc(OC)cc1"),
    _extra(['OPMP', 'PMPO'], "[O]c1ccc(OC)cc1"),
    _extra(['BocNH', 'BocHN'], "[NH]C(=O)OC(C)(C)C"),
    _extra(['BocN'], "[N]C(=O)OC(C)(C)C"),
    _extra(['NHCbz', 'CbzHN', 'CbzNH'], "[NH]C(=O)OCc1ccccc1"),
    _extra(['NHFmoc', 'FmocHN', 'FmocNH'], "[NH]C(=O)OCC1c2ccccc2-c2ccccc21"),
    _extra(['AcNH', 'AcHN'], "[NH]C(C)=O"),
    _extra(['AcO', 'OCOCH3', 'CH3CO2', 'CH3COO', 'MeCO2', 'MeCOO'], "[O]C(C)=O"),
    _extra(['AcS', 'SAc'], "[S]C(C)=O"),
    _extra(['COMe', 'MeCO', 'MeOC', 'COCH3', 'CH3CO', 'H3COC'], "[C](C)=O"),
    _extra(['PhCO'], "[C](=O)c1ccccc1"),
    _extra(['BzO', 'OCOPh', 'PhCO2', 'PhCOO'], "[O]C(=O)c1ccccc1"),

    # Sulfonyl groups and sulfonates
    _extra(['MsO'], "[O]S(C)(=O)=O"),
    _extra(['NHMs', 'MsHN', 'MsNH'], "[NH]S(C)(=O)=O"),
    _extra(['OTs', 'TsO'], "[O]S(=O)(=O)c1ccc(C)cc1"),
    _extra(['NHTs', 'TsHN', 'TsNH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),
    _extra(['NTs', 'TsN'], "[N]S(=O)(=O)c1ccc(C)cc1"),
    _extra(['TfO'], "[O]S(=O)(=O)C(F)(F)F"),
    _extra(['NHTf', 'TfHN', 'TfNH'], "[NH]S(=O)(=O)C(F)(F)F"),
    _extra(['NTf2', 'Tf2N'], "[N](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"),
    _extra(['Bs'], "[S](=O)(=O)c1ccc(Br)cc1"),
    _extra(['OBs', 'BsO'], "[O]S(=O)(=O)c1ccc(Br)cc1"),
    _extra(['Ns', 'oNs', 'o-Ns'], "[S](=O)(=O)c1ccccc1[N+](=O)[O-]"),  # Fukuyama nosyl (2-nitro)
    _extra(['pNs', 'p-Ns'], "[S](=O)(=O)c1ccc([N+](=O)[O-])cc1"),
    _extra(['SO2Me', 'MeO2S', 'MeSO2', 'SO2CH3', 'CH3SO2'], "[S](C)(=O)=O"),
    _extra(['SO2Ph', 'PhO2S', 'PhSO2'], "[S](=O)(=O)c1ccccc1"),
    _extra(['SO2NH2', 'H2NO2S', 'H2NSO2'], "[S](N)(=O)=O"),
    _extra(['SO2Cl', 'ClO2S', 'ClSO2'], "[S](Cl)(=O)=O"),
    _extra(['SO2F', 'FO2S', 'FSO2'], "[S](F)(=O)=O"),
    _extra(['HO3S'], "[S](=O)(=O)O"),
    _extra(['SCF3', 'F3CS'], "[S]C(F)(F)F"),
    _extra(['SF5', 'F5S'], "[S](F)(F)(F)(F)F"),
    _extra(['SPh', 'PhS'], "[S]c1ccccc1"),
    _extra(['SEt', 'EtS'], "[S]CC"),

    # Nitrogen groups
    _extra(['NMe2', 'Me2N'], "[N](C)C"),
    _extra(['NEt2', 'Et2N'], "[N](CC)CC"),
    _extra(['NHMe', 'MeHN', 'MeNH'], "[NH]C"),
    _extra(['NHEt', 'EtHN', 'EtNH'], "[NH]CC"),
    _extra(['NHPh', 'PhHN', 'PhNH'], "[NH]c1ccccc1"),
    _extra(['NHBn', 'BnHN', 'BnNH'], "[NH]Cc1ccccc1"),
    _extra(['NBn2', 'Bn2N'], "[N](Cc1ccccc1)Cc1ccccc1"),
    _extra(['CONH2', 'H2NOC', 'H2NCO'], "[C](N)=O"),
    _extra(['CONHMe', 'MeHNOC', 'MeNHCO', 'MeHNCO'], "[C](=O)NC"),
    _extra(['CONMe2', 'Me2NOC', 'Me2NCO'], "[C](=O)N(C)C"),
    _extra(['NHCHO', 'OHCHN'], "[NH]C=O"),
    _extra(['NHOH', 'HOHN', 'HONH'], "[NH]O"),
    _extra(['NHNH2', 'H2NHN', 'H2NNH'], "[NH]N"),
    _extra(['NNO2', 'O2NN', 'NHNO2', 'O2NHN', 'O2NNH'], "[NH][N+](=O)[O-]"),
    _extra(['NNO2-', 'O2NN-'], "[N-][N+](=O)[O-]"),
    _extra(['N3'], "[N]=[N+]=[N-]"),
    _extra(['NCO'], "[N]=C=O"),
    _extra(['NCS'], "[N]=C=S"),
    _extra(['CH2CN', 'NCCH2', 'NCH2C'], "[CH2]C#N"),

    # Carbonyl derivatives
    _extra(['HOOC'], "[C](=O)O"),
    _extra(['EtOOC'], "[C](=O)OCC"),
    _extra(['MeOOC'], "[C](=O)OC"),
    _extra(['CO2tBu', 'tBuO2C', 'COOtBu', 'tBuOOC', 'CO2But', 'ButO2C'], "[C](=O)OC(C)(C)C"),
    _extra(['CO2Bn', 'BnO2C', 'COOBn', 'BnOOC'], "[C](=O)OCc1ccccc1"),
    _extra(['CO2iPr', 'iPrO2C', 'COOiPr', 'iPrOOC'], "[C](=O)OC(C)C"),
    _extra(['CO2Ph', 'PhO2C', 'COOPh', 'PhOOC'], "[C](=O)Oc1ccccc1"),
    _extra(['CO2-', 'COO-'], "[C](=O)[O-]"),
    _extra(['COCl', 'ClOC', 'ClCO'], "[C](Cl)=O"),
    _extra(['COCF3', 'F3COC', 'CF3CO'], "[C](=O)C(F)(F)F"),
    _extra(['OCOCF3', 'CF3CO2', 'CF3COO', 'OTFA', 'TFAO'], "[O]C(=O)C(F)(F)F"),
    _extra(['CH2CO2Et', 'EtO2CCH2', 'EtO2CH2C'], "[CH2]C(=O)OCC"),
    _extra(['CH2CO2Me', 'MeO2CCH2', 'MeO2CH2C'], "[CH2]C(=O)OC"),

    # Alkyl, cycloalkyl, haloalkyl, alkoxy
    _extra(['H3C'], "[CH3]"),  # 'CH3' is left out: "[CH3]" is a valid SMILES atom and must stay one
    _extra(['Pent'], "[CH2]CCCC"),
    _extra(['Hex'], "[CH2]CCCCC"),
    _extra(['Hept', 'nHept', 'n-Hept', 'C7H15'], "[CH2]CCCCCC"),
    _extra(['Oct', 'nOct', 'n-Oct', 'C8H17'], "[CH2]CCCCCCC"),
    _extra(['cPr', 'c-Pr', 'cyPr', 'cC3H5', 'c-C3H5'], "[CH]1CC1"),
    _extra(['cBu', 'c-Bu', 'cC4H7', 'c-C4H7'], "[CH]1CCC1"),
    _extra(['cPent', 'c-Pent', 'cC5H9', 'c-C5H9'], "[CH]1CCCC1"),
    _extra(['cHex', 'c-Hex', 'cC6H11', 'c-C6H11', 'C6H11'], "[CH]1CCCCC1"),
    _extra(['sBu', 's-Bu', 'secBu', 'sec-Bu', 'Bus'], "[CH](C)CC"),
    _extra(['tert-Bu', 'But', 'Bu-t'], "[C](C)(C)C"),
    _extra(['Pri', 'Pr-i'], "[CH](C)C"),
    _extra(['Ad', '1-Ad'], "[C]12CC3CC(CC(C3)C1)C2"),
    _extra(['OiPr', 'iPrO', 'i-PrO', 'OPri', 'PriO'], "[O]C(C)C"),
    _extra(['tBuO', 't-BuO', 'ButO', 'OBut'], "[O]C(C)(C)C"),
    _extra(['OPh', 'PhO'], "[O]c1ccccc1"),
    _extra(['BnO'], "[O]Cc1ccccc1"),
    _extra(['CH2OH', 'HOH2C', 'HOCH2'], "[CH2]O"),
    _extra(['CH2OMe', 'MeOH2C', 'MeOCH2'], "[CH2]OC"),
    _extra(['CH2Ph', 'PhCH2', 'PhH2C'], "[CH2]c1ccccc1"),
    _extra(['CHF2', 'F2HC', 'CF2H', 'HF2C'], "[CH](F)F"),
    _extra(['CH2F', 'FH2C', 'FCH2'], "[CH2]F"),
    _extra(['OCHF2', 'F2HCO', 'OCF2H', 'HF2CO'], "[O]C(F)F"),
    _extra(['C2F5', 'F5C2', 'CF2CF3'], "[C](F)(F)C(F)(F)F"),
    _extra(['CBr3'], "[C](Br)(Br)Br"),

    # Aryl / heteroaryl
    _extra(['C6H5', 'H5C6'], "[c]1ccccc1"),
    _extra(['C6F5', 'F5C6'], "[c]1c(F)c(F)c(F)c(F)c1F"),
    _extra(['Tol', 'p-Tol', 'pTol', '4-Tol', 'p-tol', 'p-tolyl'], "[c]1ccc(C)cc1"),
    _extra(['o-Tol', 'oTol', '2-Tol', 'o-tol', 'o-tolyl'], "[c]1ccccc1C"),
    _extra(['m-Tol', 'mTol', '3-Tol', 'm-tol', 'm-tolyl'], "[c]1cccc(C)c1"),
    _extra(['Dipp', 'Dip'], "[c]1c(C(C)C)cccc1C(C)C"),
    _extra(['Tip', 'Trip'], "[c]1c(C(C)C)cc(C(C)C)cc1C(C)C"),
    _extra(['1-Naph', '1-Np'], "[c]1cccc2ccccc12"),
    _extra(['2-Naph', '2-Np'], "[c]1ccc2ccccc2c1"),
    _extra(['2-Py'], "[c]1ccccn1"),
    _extra(['3-Py'], "[c]1cccnc1"),
    _extra(['4-Py'], "[c]1ccncc1"),

    # Boron, tin, phosphorus
    _extra(['Bpin', 'BPin', 'pinB', 'B(pin)'], "[B]1OC(C)(C)C(C)(C)O1"),
    _extra(['B(OH)2', '(HO)2B'], "[B](O)O"),
    _extra(['BF3K', 'KF3B'], "[B-](F)(F)F.[K+]"),
    _extra(['BF3-', 'F3B-'], "[B-](F)(F)F"),
    _extra(['SnBu3', 'Bu3Sn', 'SnnBu3', 'nBu3Sn'], "[Sn](CCCC)(CCCC)CCCC"),
    _extra(['SnMe3', 'Me3Sn'], "[Sn](C)(C)C"),
    _extra(['PPh2', 'Ph2P', 'P-Ph2'], "[P](c1ccccc1)c1ccccc1"),
    _extra(['PtBu2', 'P-tBu2', 'tBu2P', 'PBut2', 'P(tBu)2'], "[P](C(C)(C)C)C(C)(C)C"),
    _extra(['PiPr2', 'P-iPr2', 'iPr2P', 'P(iPr)2'], "[P](C(C)C)C(C)C"),
    _extra(['PCy2', 'P-Cy2', 'Cy2P'], "[P](C1CCCCC1)C1CCCCC1"),
    _extra(['PMe2', 'P-Me2', 'Me2P'], "[P](C)C"),
    _extra(['P(O)Ph2', 'Ph2P(O)', 'POPh2', 'Ph2OP'], "[P](=O)(c1ccccc1)c1ccccc1"),
    _extra(['P(O)(OEt)2', '(EtO)2P(O)', 'PO(OEt)2', '(EtO)2OP', 'PO3Et2'], "[P](=O)(OCC)OCC"),
    _extra(['P(O)(OMe)2', '(MeO)2P(O)', 'PO(OMe)2', '(MeO)2OP', 'PO3Me2'], "[P](=O)(OC)OC"),
    _extra(['PPh3+', 'Ph3P+'], "[P+](c1ccccc1)(c1ccccc1)c1ccccc1"),

    # Counter-ions (drawn without bonds)
    _extra(['BF4', 'BF4-'], "F[B-](F)(F)F", 0),
    _extra(['PF6', 'PF6-'], "F[P-](F)(F)(F)(F)F", 0),
    _extra(['SbF6', 'SbF6-'], "F[Sb-](F)(F)(F)(F)F", 0),
    _extra(['AsF6', 'AsF6-'], "F[As-](F)(F)(F)(F)F", 0),
    _extra(['ClO4', 'ClO4-'], "[O-][Cl+3]([O-])([O-])[O-]", 0),
    _extra(['OTf-', 'TfO-', 'CF3SO3-', 'CF3SO3', 'OTf', 'TfO'], "[O-]S(=O)(=O)C(F)(F)F", 0),
    _extra(['NTf2-', 'Tf2N-', 'NTf2', 'Tf2N'], "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F", 0),
    _extra(['OTs-', 'TsO-', 'OTs', 'TsO'], "[O-]S(=O)(=O)c1ccc(C)cc1", 0),
    _extra(['OMs-', 'MsO-', 'OMs', 'MsO'], "[O-]S(C)(=O)=O", 0),
    _extra(['OAc-', 'AcO-', 'OAc', 'AcO'], "CC(=O)[O-]", 0),

    # In-line groups drawn with a bond on each side (Markush / patent style). Atom 0 takes the left neighbour;
    # a second radical atom, when present, takes the right one.
    _extra(['CO', 'C(O)', 'C(=O)'], "[C]=O", 2),
    _extra(['CO2', 'COO', 'C(O)O', 'C(=O)O'], "[C](=O)[O]", 2),
    _extra(['O2C', 'OOC', 'OC(O)', 'OC(=O)'], "[O][C]=O", 2),
    _extra(['CONH', 'C(O)NH', 'C(=O)NH'], "[C](=O)[NH]", 2),
    _extra(['NHCO', 'NHC(O)', 'NHC(=O)'], "[NH][C]=O", 2),
    _extra(['NHCONH', 'NHC(O)NH'], "[NH]C(=O)[NH]", 2),
    _extra(['SO2', 'S(O)2', 'S(O2)'], "[S](=O)=O", 2),
    _extra(['SO2NH', 'S(O)2NH'], "[S](=O)(=O)[NH]", 2),
    _extra(['NHSO2', 'NHS(O)2'], "[NH][S](=O)=O", 2),
    _extra(['OCH2O'], "[O]C[O]", 2),
    _extra(['CH2CH2', '(CH2)2'], "[CH2][CH2]", 2),
    _extra(['CH=CH'], "[CH]=[CH]", 2),
    _extra(['N=N'], "[N]=[N]", 2),
    # --- BEGIN GENERATED ABBREVIATIONS (benchmark/abbrev_merge.py) ---
    # N-protecting
    _extra(['(Me3Si)2N'], "[N]([Si](C)(C)C)[Si](C)(C)C"),  # bis(trimethylsilyl)amino
    _extra(['2-Br-Z', 'BrZ', '2-BrZ'], "[C](=O)OCc1ccccc1Br"),  # 2-bromobenzyloxycarbonyl
    _extra(['2-Cl-Z', 'ClZ', '2-ClZ'], "[C](=O)OCc1ccccc1Cl"),  # 2-chlorobenzyloxycarbonyl
    _extra(['2-NO2C6H4SO2HN', '2-NO2C6H4SO2NH'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['2-NO2C6H4SO2N', '2-NO2C6H4SO2-N'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['2-NO2C6H4SO2N'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['2-Ns', 'Nos'], "[S](=O)(=O)c1ccccc1[N+](=O)[O-]"),  # 2-nitrobenzenesulfonyl (o-nosyl)
    _extra(['2-O2NC6H4SO2HN', '2-O2NC6H4SO2NH'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['2-O2NC6H4SO2N', '2-O2NC6H4SO2-N'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['2-O2NC6H4SO2N'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['3,4-DMB', '3,4-Dmb', 'Veratryl', 'DMPM'], "[CH2]c1ccc(OC)c(OC)c1"),  # 3,4-dimethoxybenzyl (veratryl)
    _extra(['4-BrC6H4SO2HN', '4-BrC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-BrC6H4SO2N', '4-BrC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-BrC6H4SO2N'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['4-MeC6H4SO2HN', '4-MeC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['4-MeC6H4SO2N', '4-MeC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['4-MeC6H4SO2N'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['4-MeOC6H4SO2HN', '4-MeOC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-MeOC6H4SO2N', '4-MeOC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-MeOC6H4SO2N'], "[N]S(=O)(=O)c1ccc(OC)cc1", 2),  # N-4-methoxybenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['4-NO2C6H4SO2HN', '4-NO2C6H4SO2NH'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-NO2C6H4SO2N', '4-NO2C6H4SO2-N'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-NO2C6H4SO2N'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['4-Ns', 'Nbs'], "[S](=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # 4-nitrobenzenesulfonyl (p-nosyl)
    _extra(['4-O2NC6H4SO2HN', '4-O2NC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-O2NC6H4SO2N', '4-O2NC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['4-O2NC6H4SO2N'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['Adoc'], "[C](=O)OC12CC3CC(CC(C3)C1)C2"),  # 1-adamantyloxycarbonyl
    _extra(['AllO2CHN', 'AllO2CNH'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino, group written left
    _extra(['AllO2CN', 'AllO2C-N'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino, group written left
    _extra(['AllO2CN'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['AllO2CNBn', 'AllO2CN(Bn)'], "[N](Cc1ccccc1)C(=O)OCC=C"),  # N-benzyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['AllO2CNMe', 'AllO2CN(Me)'], "[N](C)C(=O)OCC=C"),  # N-methyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['AllylO2CHN', 'AllylO2CNH'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino, group written left
    _extra(['AllylO2CN', 'AllylO2C-N'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino, group written left
    _extra(['AllylO2CN'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['AllylO2CNBn', 'AllylO2CN(Bn)'], "[N](Cc1ccccc1)C(=O)OCC=C"),  # N-benzyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['AllylO2CNMe', 'AllylO2CN(Me)'], "[N](C)C(=O)OCC=C"),  # N-methyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['AllylOC(O)HN', 'AllylOC(O)NH'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino, group written left
    _extra(['AllylOC(O)N', 'AllylOC(O)-N'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino, group written left
    _extra(['AllylOC(O)N'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['Aloc'], "[C](=O)OCC=C"),  # allyloxycarbonyl
    _extra(['BnO2CHN', 'BnO2CNH'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['BnO2CN', 'BnO2C-N'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['BnO2CN'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['BnO2CNBn', 'BnO2CN(Bn)'], "[N](Cc1ccccc1)C(=O)OCc1ccccc1"),  # N-benzyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['BnO2CNMe', 'BnO2CN(Me)'], "[N](C)C(=O)OCc1ccccc1"),  # N-methyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['BnOC(O)HN', 'BnOC(O)NH'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['BnOC(O)N', 'BnOC(O)-N'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['BnOC(O)N'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['BnOOCHN', 'BnOOCNH'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['BnOOCN', 'BnOOC-N'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['BnOOCN'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['BnOOCNBn', 'BnOOCN(Bn)'], "[N](Cc1ccccc1)C(=O)OCc1ccccc1"),  # N-benzyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['BnOOCNMe', 'BnOOCN(Me)'], "[N](C)C(=O)OCc1ccccc1"),  # N-methyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['BOM', 'Bom'], "[CH2]OCc1ccccc1"),  # benzyloxymethyl
    _extra(['Bpoc'], "[C](=O)OC(C)(C)c1ccc(-c2ccccc2)cc1"),  # 2-(4-biphenylyl)propan-2-yloxycarbonyl
    _extra(['BrC6H4SO2HN', 'BrC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['BrC6H4SO2N', 'BrC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['BrC6H4SO2N'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['Bsmoc'], "[C](=O)OCC1=Cc2ccccc2S1(=O)=O"),  # 1,1-dioxobenzo[b]thiophen-2-ylmethoxycarbonyl
    _extra(['Bt', '1-Bt'], "[N]1N=Nc2ccccc21"),  # 1H-benzotriazol-1-yl (Katritzky aminal/amidoalkyl auxiliary)
    _extra(['ButO2CHN', 'ButO2CNH'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['ButO2CN', 'ButO2C-N'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['ButO2CN'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['Bzl'], "[CH2]c1ccccc1"),  # benzyl
    _extra(['CF3C(O)HN', 'CF3C(O)NH'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['CF3C(O)N', 'CF3C(O)-N'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['CF3C(O)N'], "[N]C(=O)C(F)(F)F", 2),  # N-trifluoroacetyl (formula spelling) nitrogen with two bonds
    _extra(['CF3COHN', 'CF3CONH'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['CF3CON', 'CF3CO-N'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['CF3CON'], "[N]C(=O)C(F)(F)F", 2),  # N-trifluoroacetyl (formula spelling) nitrogen with two bonds
    _extra(['CF3CONBn', 'CF3CON(Bn)'], "[N](Cc1ccccc1)C(=O)C(F)(F)F"),  # N-benzyl-N-(trifluoroacetyl (formula spelling))amino
    _extra(['CF3CONMe', 'CF3CON(Me)'], "[N](C)C(=O)C(F)(F)F"),  # N-methyl-N-(trifluoroacetyl (formula spelling))amino
    _extra(['CF3O2SHN', 'CF3O2SNH'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['CF3O2SN', 'CF3O2S-N'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['CF3O2SN'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['CF3S(O)2HN', 'CF3S(O)2NH'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['CF3S(O)2N', 'CF3S(O)2-N'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['CF3S(O)2N'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['CF3SO2HN', 'CF3SO2NH'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['CF3SO2N', 'CF3SO2-N'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['CF3SO2N'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['CF3SO2NBn', 'CF3SO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)C(F)(F)F"),  # N-benzyl-N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['CF3SO2NMe', 'CF3SO2N(Me)'], "[N](C)S(=O)(=O)C(F)(F)F"),  # N-methyl-N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['CH3COHN', 'CH3CONH'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['CH3CON', 'CH3CO-N'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['CH3CON'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['CH3CONBn', 'CH3CON(Bn)'], "[N](Cc1ccccc1)C(=O)C"),  # N-benzyl-N-(acetyl (formula spelling))amino
    _extra(['CH3CONMe', 'CH3CON(Me)'], "[N](C)C(=O)C"),  # N-methyl-N-(acetyl (formula spelling))amino
    _extra(['CH3SO2HN', 'CH3SO2NH'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['CH3SO2N', 'CH3SO2-N'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['CH3SO2N'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['CH3SO2NBn', 'CH3SO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)C"),  # N-benzyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['CH3SO2NMe', 'CH3SO2N(Me)'], "[N](C)S(=O)(=O)C"),  # N-methyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['Cl3CCH2O2CHN', 'Cl3CCH2O2CNH'], "[NH]C(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino, group written left
    _extra(['Cl3CCH2O2CN', 'Cl3CCH2O2C-N'], "[NH]C(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino, group written left
    _extra(['Cl3CCH2O2CN'], "[N]C(=O)OCC(Cl)(Cl)Cl", 2),  # N-2,2,2-trichloroethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['Cl3CCH2O2CNBn', 'Cl3CCH2O2CN(Bn)'], "[N](Cc1ccccc1)C(=O)OCC(Cl)(Cl)Cl"),  # N-benzyl-N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino
    _extra(['Cl3CCH2O2CNMe', 'Cl3CCH2O2CN(Me)'], "[N](C)C(=O)OCC(Cl)(Cl)Cl"),  # N-methyl-N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino
    _extra(['Cl3CCH2OC(O)HN', 'Cl3CCH2OC(O)NH'], "[NH]C(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino, group written left
    _extra(['Cl3CCH2OC(O)N', 'Cl3CCH2OC(O)-N'], "[NH]C(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino, group written left
    _extra(['Cl3CCH2OC(O)N'], "[N]C(=O)OCC(Cl)(Cl)Cl", 2),  # N-2,2,2-trichloroethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['ClAc'], "[C](=O)CCl"),  # chloroacetyl
    _extra(['ClCH2COHN', 'ClCH2CONH'], "[NH]C(=O)CCl"),  # N-(chloroacetyl (formula spelling))amino, group written left
    _extra(['ClCH2CON', 'ClCH2CO-N'], "[NH]C(=O)CCl"),  # N-(chloroacetyl (formula spelling))amino, group written left
    _extra(['ClCH2CON'], "[N]C(=O)CCl", 2),  # N-chloroacetyl (formula spelling) nitrogen with two bonds
    _extra(['ClCH2CONBn', 'ClCH2CON(Bn)'], "[N](Cc1ccccc1)C(=O)CCl"),  # N-benzyl-N-(chloroacetyl (formula spelling))amino
    _extra(['ClCH2CONMe', 'ClCH2CON(Me)'], "[N](C)C(=O)CCl"),  # N-methyl-N-(chloroacetyl (formula spelling))amino
    _extra(['ClH2CCOHN', 'ClH2CCONH'], "[NH]C(=O)CCl"),  # N-(chloroacetyl (formula spelling))amino, group written left
    _extra(['ClH2CCON', 'ClH2CCO-N'], "[NH]C(=O)CCl"),  # N-(chloroacetyl (formula spelling))amino, group written left
    _extra(['ClH2CCON'], "[N]C(=O)CCl", 2),  # N-chloroacetyl (formula spelling) nitrogen with two bonds
    _extra(['ClH2COCHN', 'ClH2COCNH'], "[NH]C(=O)CCl"),  # N-(chloroacetyl (formula spelling))amino, group written left
    _extra(['ClH2COCN', 'ClH2COC-N'], "[NH]C(=O)CCl"),  # N-(chloroacetyl (formula spelling))amino, group written left
    _extra(['ClH2COCN'], "[N]C(=O)CCl", 2),  # N-chloroacetyl (formula spelling) nitrogen with two bonds
    _extra(['Coc'], "[C](=O)OCC=Cc1ccccc1"),  # cinnamyloxycarbonyl
    _extra(['Cum', 'CMe2Ph', 'PhCMe2', 'PhMe2C', 'C(Me)2Ph', 'PhC(Me)2', 'CPhMe2'], "[C](C)(C)c1ccccc1"),  # cumyl (2-phenylpropan-2-yl)
    _extra(['Dde'], "[C](C)=C1C(=O)CC(C)(C)CC1=O"),  # 1-(4,4-dimethyl-2,6-dioxocyclohexylidene)ethyl
    _extra(['Ddz'], "[C](=O)OC(C)(C)c1cc(OC)cc(OC)c1"),  # 2-(3,5-dimethoxyphenyl)propan-2-yloxycarbonyl
    _extra(['DMB', 'Dmb', '2,4-DMB', '2,4-Dmb'], "[CH2]c1ccc(OC)cc1OC"),  # 2,4-dimethoxybenzyl
    _extra(['DMTr', 'DMT', 'Dmt'], "[C](c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1"),  # 4,4'-dimethoxytrityl
    _extra(['Dnp', 'DNP', '2,4-DNP'], "[c]1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # 2,4-dinitrophenyl
    _extra(['DNs', 'dNs', 'DNBS'], "[S](=O)(=O)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # 2,4-dinitrobenzenesulfonyl
    _extra(['Dns'], "[S](=O)(=O)c1cccc2c(N(C)C)cccc12"),  # 5-(dimethylamino)naphthalene-1-sulfonyl (dansyl)
    _extra(['Doc'], "[C](=O)OC(C(C)C)C(C)C"),  # 2,4-dimethylpent-3-yloxycarbonyl
    _extra(['Dpm', 'Ph2HC'], "[CH](c1ccccc1)c1ccccc1"),  # diphenylmethyl (benzhydryl)
    _extra(['Eoc'], "[C](=O)OCC"),  # ethoxycarbonyl
    _extra(['EtO2CHN', 'EtO2CNH'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino, group written left
    _extra(['EtO2CN', 'EtO2C-N'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino, group written left
    _extra(['EtO2CN'], "[N]C(=O)OCC", 2),  # N-ethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['EtO2CNBn', 'EtO2CN(Bn)'], "[N](Cc1ccccc1)C(=O)OCC"),  # N-benzyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['EtO2CNMe', 'EtO2CN(Me)'], "[N](C)C(=O)OCC"),  # N-methyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['EtOC(O)HN', 'EtOC(O)NH'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino, group written left
    _extra(['EtOC(O)N', 'EtOC(O)-N'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino, group written left
    _extra(['EtOC(O)N'], "[N]C(=O)OCC", 2),  # N-ethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['EtOOCHN', 'EtOOCNH'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino, group written left
    _extra(['EtOOCN', 'EtOOC-N'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino, group written left
    _extra(['EtOOCN'], "[N]C(=O)OCC", 2),  # N-ethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['EtOOCNBn', 'EtOOCN(Bn)'], "[N](Cc1ccccc1)C(=O)OCC"),  # N-benzyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['EtOOCNMe', 'EtOOCN(Me)'], "[N](C)C(=O)OCC"),  # N-methyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['F3CCOHN', 'F3CCONH'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['F3CCON', 'F3CCO-N'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['F3CCON'], "[N]C(=O)C(F)(F)F", 2),  # N-trifluoroacetyl (formula spelling) nitrogen with two bonds
    _extra(['F3CCONBn', 'F3CCON(Bn)'], "[N](Cc1ccccc1)C(=O)C(F)(F)F"),  # N-benzyl-N-(trifluoroacetyl (formula spelling))amino
    _extra(['F3CCONMe', 'F3CCON(Me)'], "[N](C)C(=O)C(F)(F)F"),  # N-methyl-N-(trifluoroacetyl (formula spelling))amino
    _extra(['F3CO2SHN', 'F3CO2SNH'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['F3CO2SN', 'F3CO2S-N'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['F3CO2SN'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['F3COCHN', 'F3COCNH'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['F3COCN', 'F3COC-N'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino, group written left
    _extra(['F3COCN'], "[N]C(=O)C(F)(F)F", 2),  # N-trifluoroacetyl (formula spelling) nitrogen with two bonds
    _extra(['F3CSO2HN', 'F3CSO2NH'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['F3CSO2N', 'F3CSO2-N'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino, group written left
    _extra(['F3CSO2N'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['F3CSO2NBn', 'F3CSO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)C(F)(F)F"),  # N-benzyl-N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['F3CSO2NMe', 'F3CSO2N(Me)'], "[N](C)S(=O)(=O)C(F)(F)F"),  # N-methyl-N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['For'], "[CH]=O"),  # formyl
    _extra(['H2N+'], "[NH2+]", 2),  # protonated secondary amine, two bonds
    _extra(['H3CO2SHN', 'H3CO2SNH'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['H3CO2SN', 'H3CO2S-N'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['H3CO2SN'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['H3COCHN', 'H3COCNH'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['H3COCN', 'H3COC-N'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['H3COCN'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['H3N+'], "[NH3+]"),  # ammonium (protonated primary amine)
    _extra(['HCOHN', 'HCONH'], "[NH]C=O"),  # N-(formyl (formula spelling))amino, group written left
    _extra(['HCON', 'HCO-N'], "[NH]C=O"),  # N-(formyl (formula spelling))amino, group written left
    _extra(['HCON'], "[N]C=O", 2),  # N-formyl (formula spelling) nitrogen with two bonds
    _extra(['HCONBn', 'HCON(Bn)'], "[N](Cc1ccccc1)C=O"),  # N-benzyl-N-(formyl (formula spelling))amino
    _extra(['HCONMe', 'HCON(Me)'], "[N](C)C=O"),  # N-methyl-N-(formyl (formula spelling))amino
    _extra(['ivDde', 'iv-Dde'], "[C](CC(C)C)=C1C(=O)CC(C)(C)CC1=O"),  # 1-(4,4-dimethyl-2,6-dioxocyclohexylidene)-3-methylbutyl
    _extra(['Mbh'], "[CH](c1ccc(OC)cc1)c1ccc(OC)cc1"),  # 4,4'-dimethoxybenzhydryl
    _extra(['Me3SiCH2CH2SO2HN', 'Me3SiCH2CH2SO2NH'], "[NH]S(=O)(=O)CC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino, group written left
    _extra(['Me3SiCH2CH2SO2N', 'Me3SiCH2CH2SO2-N'], "[NH]S(=O)(=O)CC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino, group written left
    _extra(['Me3SiCH2CH2SO2N'], "[N]S(=O)(=O)CC[Si](C)(C)C", 2),  # N-2-(trimethylsilyl)ethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['Me3SiHN', 'Me3SiNH', 'Me3SiN'], "[NH][Si](C)(C)C"),  # N-(trimethylsilyl)amino
    _extra(['Me3SiN'], "[N][Si](C)(C)C", 2),  # N-trimethylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['MeC(O)HN', 'MeC(O)NH'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['MeC(O)N', 'MeC(O)-N'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['MeC(O)N'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['MeCOHN', 'MeCONH'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['MeCON', 'MeCO-N'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['MeCON'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['MeCONBn', 'MeCON(Bn)'], "[N](Cc1ccccc1)C(=O)C"),  # N-benzyl-N-(acetyl (formula spelling))amino
    _extra(['MeCONMe', 'MeCON(Me)'], "[N](C)C(=O)C"),  # N-methyl-N-(acetyl (formula spelling))amino
    _extra(['MeO2CHN', 'MeO2CNH'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino, group written left
    _extra(['MeO2CN', 'MeO2C-N'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino, group written left
    _extra(['MeO2CN'], "[N]C(=O)OC", 2),  # N-methoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['MeO2CNBn', 'MeO2CN(Bn)'], "[N](Cc1ccccc1)C(=O)OC"),  # N-benzyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['MeO2CNMe', 'MeO2CN(Me)'], "[N](C)C(=O)OC"),  # N-methyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['MeO2SHN', 'MeO2SNH'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['MeO2SN', 'MeO2S-N'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['MeO2SN'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['MeOC(O)HN', 'MeOC(O)NH'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino, group written left
    _extra(['MeOC(O)N', 'MeOC(O)-N'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino, group written left
    _extra(['MeOC(O)N'], "[N]C(=O)OC", 2),  # N-methoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['MeOC6H4', '4-MeOC6H4', 'p-MeOC6H4', 'C6H4OMe', 'C6H4-4-OMe', 'C6H4OMe-4', 'C6H4OMe-p', 'p-An', 'pAn', 'An'], "[c]1ccc(OC)cc1"),  # 4-methoxyphenyl
    _extra(['MeOC6H4HN', 'MeOC6H4NH', 'MeOC6H4N'], "[NH]c1ccc(OC)cc1"),  # N-(4-methoxyphenyl)amino
    _extra(['MeOC6H4N'], "[N]c1ccc(OC)cc1", 2),  # N-4-methoxyphenyl nitrogen with two bonds (ring/secondary N)
    _extra(['MeOC6H4SO2HN', 'MeOC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino, group written left
    _extra(['MeOC6H4SO2N', 'MeOC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino, group written left
    _extra(['MeOC6H4SO2N'], "[N]S(=O)(=O)c1ccc(OC)cc1", 2),  # N-4-methoxybenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['MeOCHN', 'MeOCNH'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['MeOCN', 'MeOC-N'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino, group written left
    _extra(['MeOCN'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['MeOOCHN', 'MeOOCNH'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino, group written left
    _extra(['MeOOCN', 'MeOOC-N'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino, group written left
    _extra(['MeOOCN'], "[N]C(=O)OC", 2),  # N-methoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['MeOOCNBn', 'MeOOCN(Bn)'], "[N](Cc1ccccc1)C(=O)OC"),  # N-benzyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['MeOOCNMe', 'MeOOCN(Me)'], "[N](C)C(=O)OC"),  # N-methyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['MeS(O)2HN', 'MeS(O)2NH'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['MeS(O)2N', 'MeS(O)2-N'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['MeS(O)2N'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['MeSO2HN', 'MeSO2NH'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['MeSO2N', 'MeSO2-N'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino, group written left
    _extra(['MeSO2N'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['MeSO2NBn', 'MeSO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)C"),  # N-benzyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['MeSO2NMe', 'MeSO2N(Me)'], "[N](C)S(=O)(=O)C"),  # N-methyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['MesS(O)', 'MesSO', 'Mes(O)S'], "[S](=O)c1c(C)cc(C)cc1C"),  # 2,4,6-trimethylbenzenesulfinyl (mesitylsulfinyl)
    _extra(['Mmt', 'MMTr', 'MMT', 'Mmtr'], "[C](c1ccc(OC)cc1)(c1ccccc1)c1ccccc1"),  # 4-methoxytrityl
    _extra(['Moc'], "[C](=O)OC"),  # methoxycarbonyl
    _extra(['Moz', 'MeOZ'], "[C](=O)OCc1ccc(OC)cc1"),  # 4-methoxybenzyloxycarbonyl
    _extra(['MPM', 'CH2C6H4OMe', 'CH2C6H4-4-OMe', 'MeOC6H4CH2', '4-MeOC6H4CH2', 'p-MeOC6H4CH2'], "[CH2]c1ccc(OC)cc1"),  # 4-methoxybenzyl
    _extra(['Msc'], "[C](=O)OCCS(C)(=O)=O"),  # 2-(methylsulfonyl)ethoxycarbonyl
    _extra(['Mtr'], "[S](=O)(=O)c1c(C)cc(OC)c(C)c1C"),  # 4-methoxy-2,3,6-trimethylbenzenesulfonyl
    _extra(['Mts'], "[S](=O)(=O)c1c(C)cc(C)cc1C"),  # mesitylenesulfonyl (2,4,6-trimethylbenzenesulfonyl)
    _extra(['Mtt', 'Mttr'], "[C](c1ccc(C)cc1)(c1ccccc1)c1ccccc1"),  # 4-methyltrityl
    _extra(['N(Ac)2', 'NAc2', 'Ac2N', '(Ac)2N'], "[N](C(=O)C)C(=O)C"),  # bis(acetyl)amino
    _extra(['N(Ac)Allyl', 'N(Allyl)Ac', 'AcN(Allyl)', 'AllylN(Ac)', 'N(Ac)All', 'AcNAll', 'AllNAc'], "[N](CC=C)C(C)=O"),  # N-acetyl-N-allylamino
    _extra(['N(Ac)Et', 'N(Et)Ac', 'AcN(Et)', 'EtN(Ac)', 'AcNEt', 'EtNAc'], "[N](CC)C(C)=O"),  # N-acetyl-N-ethylamino
    _extra(['N(Ac)NH2', 'H2NN(Ac)', 'N(NH2)Ac', 'AcN(NH2)'], "[N](N)C(=O)C"),  # 1-(Ac)hydrazin-1-yl
    _extra(['N(Ac)Ph', 'N(Ph)Ac', 'AcN(Ph)', 'PhN(Ac)', 'AcNPh', 'PhNAc'], "[N](c1ccccc1)C(C)=O"),  # N-acetyl-N-phenylamino (acetanilide N)
    _extra(['N(Ac)PMB', 'N(PMB)Ac', 'AcN(PMB)', 'PMBN(Ac)', 'AcNPMB', 'PMBNAc'], "[N](Cc1ccc(OC)cc1)C(C)=O"),  # N-acetyl-N-(4-methoxybenzyl)amino
    _extra(['N(Alloc)2', 'NAlloc2', 'Alloc2N', '(Alloc)2N'], "[N](C(=O)OCC=C)C(=O)OCC=C"),  # bis(allyloxycarbonyl)amino
    _extra(['N(Alloc)NH2', 'H2NN(Alloc)', 'N(NH2)Alloc', 'AllocN(NH2)'], "[N](N)C(=O)OCC=C"),  # 1-(Alloc)hydrazin-1-yl
    _extra(['N(Allyl)2', 'NAllyl2', 'Allyl2N', '(Allyl)2N', 'N(All)2', 'NAll2', 'All2N', '(All)2N', 'N(CH2CH=CH2)2'], "[N](CC=C)CC=C"),  # diallylamino
    _extra(['N(Bn)2', '(Bn)2N'], "[N](Cc1ccccc1)Cc1ccccc1"),  # bis(benzyl)amino
    _extra(['N(Bn)Ac', 'NBnAc', 'N(Ac)Bn', 'AcNBn', 'AcN(Bn)', 'BnNAc', 'BnN(Ac)'], "[N](Cc1ccccc1)C(=O)C"),  # N-benzyl-N-(acetyl)amino
    _extra(['N(Bn)Alloc', 'NBnAlloc', 'N(Alloc)Bn', 'AllocNBn', 'AllocN(Bn)', 'BnNAlloc', 'BnN(Alloc)'], "[N](Cc1ccccc1)C(=O)OCC=C"),  # N-benzyl-N-(allyloxycarbonyl)amino
    _extra(['N(Bn)Allyl', 'N(Allyl)Bn', 'BnN(Allyl)', 'AllylN(Bn)', 'N(Bn)All', 'BnN(All)', 'AllN(Bn)', 'BnNAll', 'AllNBn'], "[N](CC=C)Cc1ccccc1"),  # N-allyl-N-benzylamino
    _extra(['N(Bn)Boc', 'NBnBoc', 'N(Boc)Bn', 'BocNBn', 'BocN(Bn)', 'BnNBoc', 'BnN(Boc)'], "[N](Cc1ccccc1)C(=O)OC(C)(C)C"),  # N-benzyl-N-(tert-butoxycarbonyl)amino
    _extra(['N(Bn)Bz', 'NBnBz', 'N(Bz)Bn', 'BzNBn', 'BzN(Bn)', 'BnNBz', 'BnN(Bz)'], "[N](Cc1ccccc1)C(=O)c1ccccc1"),  # N-benzyl-N-(benzoyl)amino
    _extra(['N(Bn)Cbz', 'NBnCbz', 'N(Cbz)Bn', 'CbzNBn', 'CbzN(Bn)', 'BnNCbz', 'BnN(Cbz)'], "[N](Cc1ccccc1)C(=O)OCc1ccccc1"),  # N-benzyl-N-(benzyloxycarbonyl)amino
    _extra(['N(Bn)CHO', 'NBnCHO', 'BnNCHO'], "[N](Cc1ccccc1)C=O"),  # N-benzyl-N-(formyl (formula spelling))amino
    _extra(['N(Bn)CO2All', 'NBnCO2All', 'BnNCO2All'], "[N](Cc1ccccc1)C(=O)OCC=C"),  # N-benzyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['N(Bn)CO2Allyl', 'NBnCO2Allyl', 'BnNCO2Allyl'], "[N](Cc1ccccc1)C(=O)OCC=C"),  # N-benzyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['N(Bn)CO2Bn', 'NBnCO2Bn', 'BnNCO2Bn'], "[N](Cc1ccccc1)C(=O)OCc1ccccc1"),  # N-benzyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['N(Bn)CO2CH2CCl3', 'NBnCO2CH2CCl3', 'BnNCO2CH2CCl3'], "[N](Cc1ccccc1)C(=O)OCC(Cl)(Cl)Cl"),  # N-benzyl-N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino
    _extra(['N(Bn)CO2Et', 'NBnCO2Et', 'BnNCO2Et'], "[N](Cc1ccccc1)C(=O)OCC"),  # N-benzyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['N(Bn)CO2Me', 'NBnCO2Me', 'BnNCO2Me'], "[N](Cc1ccccc1)C(=O)OC"),  # N-benzyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['N(Bn)CO2tBu', 'NBnCO2tBu', 'BnNCO2tBu'], "[N](Cc1ccccc1)C(=O)OC(C)(C)C"),  # N-benzyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['N(Bn)COCF3', 'NBnCOCF3', 'BnNCOCF3'], "[N](Cc1ccccc1)C(=O)C(F)(F)F"),  # N-benzyl-N-(trifluoroacetyl (formula spelling))amino
    _extra(['N(Bn)COCH2Cl', 'NBnCOCH2Cl', 'BnNCOCH2Cl'], "[N](Cc1ccccc1)C(=O)CCl"),  # N-benzyl-N-(chloroacetyl (formula spelling))amino
    _extra(['N(Bn)COCH3', 'NBnCOCH3', 'BnNCOCH3'], "[N](Cc1ccccc1)C(=O)C"),  # N-benzyl-N-(acetyl (formula spelling))amino
    _extra(['N(Bn)COMe', 'NBnCOMe', 'BnNCOMe'], "[N](Cc1ccccc1)C(=O)C"),  # N-benzyl-N-(acetyl (formula spelling))amino
    _extra(['N(Bn)COOBn', 'NBnCOOBn', 'BnNCOOBn'], "[N](Cc1ccccc1)C(=O)OCc1ccccc1"),  # N-benzyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['N(Bn)COOEt', 'NBnCOOEt', 'BnNCOOEt'], "[N](Cc1ccccc1)C(=O)OCC"),  # N-benzyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['N(Bn)COOMe', 'NBnCOOMe', 'BnNCOOMe'], "[N](Cc1ccccc1)C(=O)OC"),  # N-benzyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['N(Bn)COOtBu', 'NBnCOOtBu', 'BnNCOOtBu'], "[N](Cc1ccccc1)C(=O)OC(C)(C)C"),  # N-benzyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['N(Bn)COPh', 'NBnCOPh', 'BnNCOPh'], "[N](Cc1ccccc1)C(=O)c1ccccc1"),  # N-benzyl-N-(benzoyl (formula spelling))amino
    _extra(['N(Bn)COtBu', 'NBnCOtBu', 'BnNCOtBu'], "[N](Cc1ccccc1)C(=O)C(C)(C)C"),  # N-benzyl-N-(pivaloyl (formula spelling))amino
    _extra(['N(Bn)Et', 'N(Et)Bn', 'BnN(Et)', 'EtN(Bn)', 'BnNEt', 'EtNBn'], "[N](CC)Cc1ccccc1"),  # N-benzyl-N-ethylamino
    _extra(['N(Bn)Fmoc', 'NBnFmoc', 'N(Fmoc)Bn', 'FmocNBn', 'FmocN(Bn)', 'BnNFmoc', 'BnN(Fmoc)'], "[N](Cc1ccccc1)C(=O)OCC1c2ccccc2-c2ccccc21"),  # N-benzyl-N-(9-fluorenylmethoxycarbonyl)amino
    _extra(['N(Bn)iPr', 'N(iPr)Bn', 'BnN(iPr)', 'iPrN(Bn)'], "[N](C(C)C)Cc1ccccc1"),  # N-benzyl-N-isopropylamino
    _extra(['N(Bn)Ms', 'NBnMs', 'N(Ms)Bn', 'MsNBn', 'MsN(Bn)', 'BnNMs', 'BnN(Ms)'], "[N](Cc1ccccc1)S(=O)(=O)C"),  # N-benzyl-N-(methanesulfonyl (mesyl))amino
    _extra(['N(Bn)NH2', 'H2NN(Bn)', 'N(NH2)Bn', 'BnN(NH2)'], "[N](N)Cc1ccccc1"),  # 1-(Bn)hydrazin-1-yl
    _extra(['N(Bn)Ns', 'NBnNs', 'N(Ns)Bn', 'NsNBn', 'NsN(Bn)', 'BnNNs', 'BnN(Ns)'], "[N](Cc1ccccc1)S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-benzyl-N-(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['N(Bn)OBn', 'BnN(OBn)', 'BnONBn', 'N(OBn)Bn', 'BnON(Bn)'], "[N](Cc1ccccc1)OCc1ccccc1"),  # N-benzyl-N-(benzyloxy)amino
    _extra(['N(Bn)Ph', 'N(Ph)Bn', 'BnN(Ph)', 'PhN(Bn)', 'BnNPh', 'PhNBn'], "[N](c1ccccc1)Cc1ccccc1"),  # N-benzyl-N-phenylamino
    _extra(['N(Bn)PMB', 'NBnPMB', 'N(PMB)Bn', 'PMBNBn', 'PMBN(Bn)', 'BnNPMB', 'BnN(PMB)'], "[N](Cc1ccccc1)Cc1ccc(OC)cc1"),  # N-benzyl-N-(4-methoxybenzyl)amino
    _extra(['N(Bn)SES', 'NBnSES', 'N(SES)Bn', 'SESNBn', 'SESN(Bn)', 'BnNSES', 'BnN(SES)'], "[N](Cc1ccccc1)S(=O)(=O)CC[Si](C)(C)C"),  # N-benzyl-N-(2-(trimethylsilyl)ethanesulfonyl)amino
    _extra(['N(Bn)SO2CF3', 'NBnSO2CF3', 'BnNSO2CF3'], "[N](Cc1ccccc1)S(=O)(=O)C(F)(F)F"),  # N-benzyl-N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['N(Bn)SO2CH2CH2TMS', 'NBnSO2CH2CH2TMS', 'BnNSO2CH2CH2TMS'], "[N](Cc1ccccc1)S(=O)(=O)CC[Si](C)(C)C"),  # N-benzyl-N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino
    _extra(['N(Bn)SO2CH3', 'NBnSO2CH3', 'BnNSO2CH3'], "[N](Cc1ccccc1)S(=O)(=O)C"),  # N-benzyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['N(Bn)SO2Me', 'NBnSO2Me', 'BnNSO2Me'], "[N](Cc1ccccc1)S(=O)(=O)C"),  # N-benzyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['N(Bn)SO2p-Tol', 'NBnSO2p-Tol', 'BnNSO2p-Tol'], "[N](Cc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-benzyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(Bn)SO2Ph', 'NBnSO2Ph', 'BnNSO2Ph'], "[N](Cc1ccccc1)S(=O)(=O)c1ccccc1"),  # N-benzyl-N-(benzenesulfonyl (formula spelling))amino
    _extra(['N(Bn)SO2pTol', 'NBnSO2pTol', 'BnNSO2pTol'], "[N](Cc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-benzyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(Bn)SO2Tol', 'NBnSO2Tol', 'BnNSO2Tol'], "[N](Cc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-benzyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(Bn)Tf', 'NBnTf', 'N(Tf)Bn', 'TfNBn', 'TfN(Bn)', 'BnNTf', 'BnN(Tf)'], "[N](Cc1ccccc1)S(=O)(=O)C(F)(F)F"),  # N-benzyl-N-(trifluoromethanesulfonyl (triflyl))amino
    _extra(['N(Bn)Ts', 'NBnTs', 'N(Ts)Bn', 'TsNBn', 'TsN(Bn)', 'BnNTs', 'BnN(Ts)'], "[N](Cc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-benzyl-N-(p-toluenesulfonyl (tosyl))amino
    _extra(['N(Bn)Z', 'NBnZ', 'N(Z)Bn', 'ZNBn', 'ZN(Bn)', 'BnNZ', 'BnN(Z)'], "[N](Cc1ccccc1)C(=O)OCc1ccccc1"),  # N-benzyl-N-(benzyloxycarbonyl)amino
    _extra(['N(Boc)2', 'NBoc2', 'Boc2N', '(Boc)2N'], "[N](C(=O)OC(C)(C)C)C(=O)OC(C)(C)C"),  # bis(tert-butoxycarbonyl)amino
    _extra(['N(Boc)Ac', 'N(Ac)Boc', 'BocN(Ac)', 'AcN(Boc)', 'BocNAc', 'AcNBoc'], "[N](C(=O)OC(C)(C)C)C(C)=O"),  # N-acetyl-N-Boc-amino
    _extra(['N(Boc)Alloc', 'N(Alloc)Boc', 'BocN(Alloc)', 'AllocN(Boc)'], "[N](C(=O)OC(C)(C)C)C(=O)OCC=C"),  # N-Alloc-N-Boc-amino
    _extra(['N(Boc)Allyl', 'N(Allyl)Boc', 'BocN(Allyl)', 'AllylN(Boc)', 'N(Boc)All', 'N(All)Boc', 'BocN(All)', 'AllN(Boc)', 'BocNAll', 'AllNBoc'], "[N](CC=C)C(=O)OC(C)(C)C"),  # N-allyl-N-Boc-amino
    _extra(['N(Boc)Bz', 'N(Bz)Boc', 'BocN(Bz)', 'BzN(Boc)', 'BocNBz', 'BzNBoc'], "[N](C(=O)OC(C)(C)C)C(=O)c1ccccc1"),  # N-benzoyl-N-Boc-amino
    _extra(['N(Boc)Cbz', 'N(Cbz)Boc', 'BocN(Cbz)', 'CbzN(Boc)', 'BocNCbz', 'CbzNBoc', 'N(Boc)Z', 'N(Z)Boc'], "[N](C(=O)OC(C)(C)C)C(=O)OCc1ccccc1"),  # N-Boc-N-Cbz-amino
    _extra(['N(Boc)CH2CO2Et', 'N(CH2CO2Et)Boc', 'BocN(CH2CO2Et)', 'EtO2CCH2N(Boc)'], "[N](CC(=O)OCC)C(=O)OC(C)(C)C"),  # N-Boc-N-(ethoxycarbonylmethyl)amino
    _extra(['N(Boc)CH2CO2Me', 'N(CH2CO2Me)Boc', 'BocN(CH2CO2Me)', 'MeO2CCH2N(Boc)'], "[N](CC(=O)OC)C(=O)OC(C)(C)C"),  # N-Boc-N-(methoxycarbonylmethyl)amino
    _extra(['N(Boc)CH2CO2tBu', 'N(CH2CO2tBu)Boc', 'BocN(CH2CO2tBu)', 'tBuO2CCH2N(Boc)'], "[N](CC(=O)OC(C)(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-(tert-butoxycarbonylmethyl)amino
    _extra(['N(Boc)Et', 'N(Et)Boc', 'BocN(Et)', 'EtN(Boc)', 'BocNEt', 'EtNBoc', 'NEtBoc'], "[N](CC)C(=O)OC(C)(C)C"),  # N-Boc-N-ethylamino
    _extra(['N(Boc)Fmoc', 'N(Fmoc)Boc', 'BocN(Fmoc)', 'FmocN(Boc)'], "[N](C(=O)OC(C)(C)C)C(=O)OCC1c2ccccc2-c2ccccc21"),  # N-Boc-N-Fmoc-amino
    _extra(['N(Boc)iPr', 'N(iPr)Boc', 'BocN(iPr)', 'iPrN(Boc)', 'BocNiPr', 'iPrNBoc', 'N(Boc)i-Pr'], "[N](C(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-isopropylamino
    _extra(['N(Boc)Ms', 'N(Ms)Boc', 'BocN(Ms)', 'MsN(Boc)', 'BocNMs', 'MsNBoc'], "[N](C(=O)OC(C)(C)C)S(C)(=O)=O"),  # N-Boc-N-mesylamino
    _extra(['N(Boc)N(Boc)Me', 'MeN(Boc)N(Boc)'], "[N](N(C)C(=O)OC(C)(C)C)C(=O)OC(C)(C)C"),  # 1,2-bis(Boc)-2-methylhydrazin-1-yl
    _extra(['N(Boc)NH2', 'H2NN(Boc)', 'N(NH2)Boc', 'BocN(NH2)'], "[N](N)C(=O)OC(C)(C)C"),  # 1-(Boc)hydrazin-1-yl
    _extra(['N(Boc)NHBoc', 'BocN(NHBoc)', 'BocNHN(Boc)', 'NBocNHBoc', 'BocHNN(Boc)'], "[N](NC(=O)OC(C)(C)C)C(=O)OC(C)(C)C"),  # 1,2-bis(tert-butoxycarbonyl)hydrazin-1-yl
    _extra(['N(Boc)NHCbz', 'BocN(NHCbz)', 'N(Boc)NHZ'], "[N](NC(=O)OCc1ccccc1)C(=O)OC(C)(C)C"),  # 1-Boc-2-Cbz-hydrazin-1-yl
    _extra(['N(Boc)Ns', 'N(Ns)Boc', 'BocN(Ns)', 'NsN(Boc)', 'BocNNs', 'NsNBoc'], "[N](C(=O)OC(C)(C)C)S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-Boc-N-nosylamino
    _extra(['N(Boc)OAc', 'N(OAc)Boc', 'BocN(OAc)', 'AcON(Boc)'], "[N](OC(C)=O)C(=O)OC(C)(C)C"),  # N-acetoxy-N-Boc-amino
    _extra(['N(Boc)OBz', 'N(OBz)Boc', 'BocN(OBz)', 'BzON(Boc)'], "[N](OC(=O)c1ccccc1)C(=O)OC(C)(C)C"),  # N-(benzoyloxy)-N-Boc-amino
    _extra(['N(Boc)OPiv', 'N(OPiv)Boc', 'BocN(OPiv)', 'PivON(Boc)'], "[N](OC(=O)C(C)(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-(pivaloyloxy)amino
    _extra(['N(Boc)OTBS', 'N(OTBS)Boc', 'BocN(OTBS)', 'TBSON(Boc)', 'TBSONBoc'], "[N](O[Si](C)(C)C(C)(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-(TBS-oxy)amino
    _extra(['N(Boc)OtBu', 'N(OtBu)Boc', 'BocN(OtBu)', 'tBuON(Boc)'], "[N](OC(C)(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-(tert-butoxy)amino
    _extra(['N(Boc)OTHP', 'N(OTHP)Boc', 'BocN(OTHP)', 'THPON(Boc)'], "[N](OC1CCCCO1)C(=O)OC(C)(C)C"),  # N-Boc-N-(THP-oxy)amino
    _extra(['N(Boc)Ph', 'N(Ph)Boc', 'BocN(Ph)', 'PhN(Boc)', 'BocNPh', 'PhNBoc'], "[N](c1ccccc1)C(=O)OC(C)(C)C"),  # N-Boc-N-phenylamino
    _extra(['N(Boc)PMB', 'N(PMB)Boc', 'BocN(PMB)', 'PMBN(Boc)', 'BocNPMB', 'PMBNBoc', 'NPMBBoc'], "[N](Cc1ccc(OC)cc1)C(=O)OC(C)(C)C"),  # N-Boc-N-(4-methoxybenzyl)amino
    _extra(['N(Boc)PMP', 'N(PMP)Boc', 'BocN(PMP)', 'PMPN(Boc)', 'BocNPMP', 'PMPNBoc'], "[N](c1ccc(OC)cc1)C(=O)OC(C)(C)C"),  # N-Boc-N-(4-methoxyphenyl)amino
    _extra(['N(Boc)tBu', 'N(tBu)Boc', 'BocN(tBu)', 'tBuN(Boc)', 'N(Boc)t-Bu'], "[N](C(C)(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-tert-butylamino
    _extra(['N(Boc)Tf', 'N(Tf)Boc', 'BocN(Tf)', 'TfN(Boc)', 'BocNTf', 'TfNBoc'], "[N](C(=O)OC(C)(C)C)S(=O)(=O)C(F)(F)F"),  # N-Boc-N-triflylamino
    _extra(['N(Boc)TMS', 'N(TMS)Boc', 'BocN(TMS)', 'TMSN(Boc)', 'N(Boc)SiMe3', 'Me3SiN(Boc)'], "[N]([Si](C)(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-(trimethylsilyl)amino
    _extra(['N(Boc)Ts', 'N(Ts)Boc', 'BocN(Ts)', 'TsN(Boc)', 'BocNTs', 'TsNBoc'], "[N](C(=O)OC(C)(C)C)S(=O)(=O)c1ccc(C)cc1"),  # N-Boc-N-tosylamino
    _extra(['N(Bz)2', 'NBz2', 'Bz2N', '(Bz)2N'], "[N](C(=O)c1ccccc1)C(=O)c1ccccc1"),  # bis(benzoyl)amino
    _extra(['N(Bz)NH2', 'H2NN(Bz)', 'N(NH2)Bz', 'BzN(NH2)'], "[N](N)C(=O)c1ccccc1"),  # 1-(Bz)hydrazin-1-yl
    _extra(['N(Bz)Ph', 'N(Ph)Bz', 'BzN(Ph)', 'PhN(Bz)', 'BzNPh', 'PhNBz'], "[N](c1ccccc1)C(=O)c1ccccc1"),  # N-benzoyl-N-phenylamino
    _extra(['N(Cbz)2', 'NCbz2', 'Cbz2N', '(Cbz)2N'], "[N](C(=O)OCc1ccccc1)C(=O)OCc1ccccc1"),  # bis(benzyloxycarbonyl)amino
    _extra(['N(Cbz)Ac', 'N(Ac)Cbz', 'CbzN(Ac)', 'AcN(Cbz)'], "[N](C(=O)OCc1ccccc1)C(C)=O"),  # N-acetyl-N-Cbz-amino
    _extra(['N(Cbz)Allyl', 'N(Allyl)Cbz', 'CbzN(Allyl)', 'AllylN(Cbz)', 'N(Cbz)All', 'CbzN(All)', 'AllN(Cbz)'], "[N](CC=C)C(=O)OCc1ccccc1"),  # N-allyl-N-Cbz-amino
    _extra(['N(Cbz)Et', 'N(Et)Cbz', 'CbzN(Et)', 'EtN(Cbz)'], "[N](CC)C(=O)OCc1ccccc1"),  # N-Cbz-N-ethylamino
    _extra(['N(Cbz)NH2', 'H2NN(Cbz)', 'N(NH2)Cbz', 'CbzN(NH2)'], "[N](N)C(=O)OCc1ccccc1"),  # 1-(Cbz)hydrazin-1-yl
    _extra(['N(Cbz)NHBoc', 'CbzN(NHBoc)', 'N(Z)NHBoc'], "[N](NC(=O)OC(C)(C)C)C(=O)OCc1ccccc1"),  # 1-Cbz-2-Boc-hydrazin-1-yl
    _extra(['N(Cbz)NHCbz', 'CbzN(NHCbz)', 'CbzNHN(Cbz)', 'N(Z)NHZ', 'ZN(NHZ)'], "[N](NC(=O)OCc1ccccc1)C(=O)OCc1ccccc1"),  # 1,2-bis(benzyloxycarbonyl)hydrazin-1-yl
    _extra(['N(Cbz)Ph', 'N(Ph)Cbz', 'CbzN(Ph)', 'PhN(Cbz)'], "[N](c1ccccc1)C(=O)OCc1ccccc1"),  # N-Cbz-N-phenylamino
    _extra(['N(Cbz)PMB', 'N(PMB)Cbz', 'CbzN(PMB)', 'PMBN(Cbz)', 'CbzNPMB', 'PMBNCbz'], "[N](Cc1ccc(OC)cc1)C(=O)OCc1ccccc1"),  # N-Cbz-N-(4-methoxybenzyl)amino
    _extra(['N(Cbz)Ts', 'N(Ts)Cbz', 'CbzN(Ts)', 'TsN(Cbz)', 'CbzNTs', 'TsNCbz'], "[N](C(=O)OCc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-Cbz-N-tosylamino
    _extra(['N(CHO)2', '(CHO)2N'], "[N](C=O)C=O"),  # bis(formyl (formula spelling))amino
    _extra(['N(CO2All)2', '(CO2All)2N'], "[N](C(=O)OCC=C)C(=O)OCC=C"),  # bis(allyloxycarbonyl (formula spelling))amino
    _extra(['N(CO2Allyl)2', '(CO2Allyl)2N'], "[N](C(=O)OCC=C)C(=O)OCC=C"),  # bis(allyloxycarbonyl (formula spelling))amino
    _extra(['N(CO2Bn)2', '(CO2Bn)2N'], "[N](C(=O)OCc1ccccc1)C(=O)OCc1ccccc1"),  # bis(benzyloxycarbonyl (formula spelling))amino
    _extra(['N(CO2CH2CCl3)2', '(CO2CH2CCl3)2N'], "[N](C(=O)OCC(Cl)(Cl)Cl)C(=O)OCC(Cl)(Cl)Cl"),  # bis(2,2,2-trichloroethoxycarbonyl (formula spelling))amino
    _extra(['N(CO2Et)2', '(CO2Et)2N'], "[N](C(=O)OCC)C(=O)OCC"),  # bis(ethoxycarbonyl (formula spelling))amino
    _extra(['N(CO2Me)2', '(CO2Me)2N'], "[N](C(=O)OC)C(=O)OC"),  # bis(methoxycarbonyl (formula spelling))amino
    _extra(['N(CO2tBu)2', '(CO2tBu)2N'], "[N](C(=O)OC(C)(C)C)C(=O)OC(C)(C)C"),  # bis(tert-butoxycarbonyl (formula spelling))amino
    _extra(['N(COCF3)2', '(COCF3)2N'], "[N](C(=O)C(F)(F)F)C(=O)C(F)(F)F"),  # bis(trifluoroacetyl (formula spelling))amino
    _extra(['N(COCH2Cl)2', '(COCH2Cl)2N'], "[N](C(=O)CCl)C(=O)CCl"),  # bis(chloroacetyl (formula spelling))amino
    _extra(['N(COCH3)2', '(COCH3)2N'], "[N](C(=O)C)C(=O)C"),  # bis(acetyl (formula spelling))amino
    _extra(['N(COCH3)Me'], "[N](C)C(=O)C"),  # N-methyl-N-(acetyl (formula spelling))amino
    _extra(['N(COMe)2', '(COMe)2N'], "[N](C(=O)C)C(=O)C"),  # bis(acetyl (formula spelling))amino
    _extra(['N(COOBn)2', '(COOBn)2N'], "[N](C(=O)OCc1ccccc1)C(=O)OCc1ccccc1"),  # bis(benzyloxycarbonyl (formula spelling))amino
    _extra(['N(COOEt)2', '(COOEt)2N'], "[N](C(=O)OCC)C(=O)OCC"),  # bis(ethoxycarbonyl (formula spelling))amino
    _extra(['N(COOMe)2', '(COOMe)2N'], "[N](C(=O)OC)C(=O)OC"),  # bis(methoxycarbonyl (formula spelling))amino
    _extra(['N(COOtBu)2', '(COOtBu)2N'], "[N](C(=O)OC(C)(C)C)C(=O)OC(C)(C)C"),  # bis(tert-butoxycarbonyl (formula spelling))amino
    _extra(['N(COPh)2', '(COPh)2N'], "[N](C(=O)c1ccccc1)C(=O)c1ccccc1"),  # bis(benzoyl (formula spelling))amino
    _extra(['N(COtBu)2', '(COtBu)2N'], "[N](C(=O)C(C)(C)C)C(=O)C(C)(C)C"),  # bis(pivaloyl (formula spelling))amino
    _extra(['N(Fmoc)2', 'NFmoc2', 'Fmoc2N', '(Fmoc)2N'], "[N](C(=O)OCC1c2ccccc2-c2ccccc21)C(=O)OCC1c2ccccc2-c2ccccc21"),  # bis(9-fluorenylmethoxycarbonyl)amino
    _extra(['N(Fmoc)Allyl', 'N(Allyl)Fmoc', 'FmocN(Allyl)', 'AllylN(Fmoc)'], "[N](CC=C)C(=O)OCC1c2ccccc2-c2ccccc21"),  # N-allyl-N-Fmoc-amino
    _extra(['N(Fmoc)NH2', 'H2NN(Fmoc)', 'N(NH2)Fmoc', 'FmocN(NH2)'], "[N](N)C(=O)OCC1c2ccccc2-c2ccccc21"),  # 1-(Fmoc)hydrazin-1-yl
    _extra(['N(Fmoc)PMB', 'N(PMB)Fmoc', 'FmocN(PMB)', 'PMBN(Fmoc)'], "[N](Cc1ccc(OC)cc1)C(=O)OCC1c2ccccc2-c2ccccc21"),  # N-Fmoc-N-(4-methoxybenzyl)amino
    _extra(['N(Me)Ac', 'NMeAc', 'N(Ac)Me', 'AcNMe', 'AcN(Me)', 'MeNAc', 'MeN(Ac)'], "[N](C)C(=O)C"),  # N-methyl-N-(acetyl)amino
    _extra(['N(Me)Alloc', 'NMeAlloc', 'N(Alloc)Me', 'AllocNMe', 'AllocN(Me)', 'MeNAlloc', 'MeN(Alloc)'], "[N](C)C(=O)OCC=C"),  # N-methyl-N-(allyloxycarbonyl)amino
    _extra(['N(Me)Bn', 'NMeBn', 'N(Bn)Me', 'BnNMe', 'BnN(Me)', 'MeNBn', 'MeN(Bn)'], "[N](C)Cc1ccccc1"),  # N-methyl-N-(benzyl)amino
    _extra(['N(Me)Boc', 'NMeBoc', 'N(Boc)Me', 'BocNMe', 'BocN(Me)', 'MeNBoc', 'MeN(Boc)'], "[N](C)C(=O)OC(C)(C)C"),  # N-methyl-N-(tert-butoxycarbonyl)amino
    _extra(['N(Me)BOM', 'NMeBOM', 'N(BOM)Me', 'BOMNMe', 'BOMN(Me)', 'MeNBOM', 'MeN(BOM)'], "[N](C)COCc1ccccc1"),  # N-methyl-N-(benzyloxymethyl)amino
    _extra(['N(Me)Bz', 'NMeBz', 'N(Bz)Me', 'BzNMe', 'BzN(Me)', 'MeNBz', 'MeN(Bz)'], "[N](C)C(=O)c1ccccc1"),  # N-methyl-N-(benzoyl)amino
    _extra(['N(Me)Cbz', 'NMeCbz', 'N(Cbz)Me', 'CbzNMe', 'CbzN(Me)', 'MeNCbz', 'MeN(Cbz)'], "[N](C)C(=O)OCc1ccccc1"),  # N-methyl-N-(benzyloxycarbonyl)amino
    _extra(['N(Me)CHO', 'NMeCHO', 'MeNCHO', 'N(CHO)Me'], "[N](C)C=O"),  # N-methyl-N-(formyl (formula spelling))amino
    _extra(['N(Me)CO2All', 'NMeCO2All', 'MeNCO2All', 'N(CO2All)Me'], "[N](C)C(=O)OCC=C"),  # N-methyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['N(Me)CO2Allyl', 'NMeCO2Allyl', 'MeNCO2Allyl', 'N(CO2Allyl)Me'], "[N](C)C(=O)OCC=C"),  # N-methyl-N-(allyloxycarbonyl (formula spelling))amino
    _extra(['N(Me)CO2Bn', 'NMeCO2Bn', 'MeNCO2Bn', 'N(CO2Bn)Me'], "[N](C)C(=O)OCc1ccccc1"),  # N-methyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['N(Me)CO2CH2CCl3', 'NMeCO2CH2CCl3', 'MeNCO2CH2CCl3', 'N(CO2CH2CCl3)Me'], "[N](C)C(=O)OCC(Cl)(Cl)Cl"),  # N-methyl-N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino
    _extra(['N(Me)CO2Et', 'NMeCO2Et', 'MeNCO2Et', 'N(CO2Et)Me'], "[N](C)C(=O)OCC"),  # N-methyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['N(Me)CO2Me', 'NMeCO2Me', 'MeNCO2Me', 'N(CO2Me)Me'], "[N](C)C(=O)OC"),  # N-methyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['N(Me)CO2tBu', 'NMeCO2tBu', 'MeNCO2tBu', 'N(CO2tBu)Me'], "[N](C)C(=O)OC(C)(C)C"),  # N-methyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['N(Me)COCF3', 'NMeCOCF3', 'MeNCOCF3', 'N(COCF3)Me'], "[N](C)C(=O)C(F)(F)F"),  # N-methyl-N-(trifluoroacetyl (formula spelling))amino
    _extra(['N(Me)COCH2Cl', 'NMeCOCH2Cl', 'MeNCOCH2Cl', 'N(COCH2Cl)Me'], "[N](C)C(=O)CCl"),  # N-methyl-N-(chloroacetyl (formula spelling))amino
    _extra(['N(Me)COMe', 'NMeCOMe', 'MeNCOMe', 'N(COMe)Me'], "[N](C)C(=O)C"),  # N-methyl-N-(acetyl (formula spelling))amino
    _extra(['N(Me)COOBn', 'NMeCOOBn', 'MeNCOOBn', 'N(COOBn)Me'], "[N](C)C(=O)OCc1ccccc1"),  # N-methyl-N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['N(Me)COOEt', 'NMeCOOEt', 'MeNCOOEt', 'N(COOEt)Me'], "[N](C)C(=O)OCC"),  # N-methyl-N-(ethoxycarbonyl (formula spelling))amino
    _extra(['N(Me)COOMe', 'NMeCOOMe', 'MeNCOOMe', 'N(COOMe)Me'], "[N](C)C(=O)OC"),  # N-methyl-N-(methoxycarbonyl (formula spelling))amino
    _extra(['N(Me)COOtBu', 'NMeCOOtBu', 'MeNCOOtBu', 'N(COOtBu)Me'], "[N](C)C(=O)OC(C)(C)C"),  # N-methyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['N(Me)COPh', 'NMeCOPh', 'MeNCOPh', 'N(COPh)Me'], "[N](C)C(=O)c1ccccc1"),  # N-methyl-N-(benzoyl (formula spelling))amino
    _extra(['N(Me)COtBu', 'NMeCOtBu', 'MeNCOtBu', 'N(COtBu)Me'], "[N](C)C(=O)C(C)(C)C"),  # N-methyl-N-(pivaloyl (formula spelling))amino
    _extra(['N(Me)Dnp', 'NMeDnp', 'N(Dnp)Me', 'DnpNMe', 'DnpN(Me)', 'MeNDnp', 'MeN(Dnp)'], "[N](C)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # N-methyl-N-(2,4-dinitrophenyl)amino
    _extra(['N(Me)Fmoc', 'NMeFmoc', 'N(Fmoc)Me', 'FmocNMe', 'FmocN(Me)', 'MeNFmoc', 'MeN(Fmoc)'], "[N](C)C(=O)OCC1c2ccccc2-c2ccccc21"),  # N-methyl-N-(9-fluorenylmethoxycarbonyl)amino
    _extra(['N(Me)Moc', 'NMeMoc', 'N(Moc)Me', 'MocNMe', 'MocN(Me)', 'MeNMoc', 'MeN(Moc)'], "[N](C)C(=O)OC"),  # N-methyl-N-(methoxycarbonyl)amino
    _extra(['N(Me)MOM', 'NMeMOM', 'N(MOM)Me', 'MOMNMe', 'MOMN(Me)', 'MeNMOM', 'MeN(MOM)'], "[N](C)COC"),  # N-methyl-N-(methoxymethyl)amino
    _extra(['N(Me)Ms', 'NMeMs', 'N(Ms)Me', 'MsNMe', 'MsN(Me)', 'MeNMs', 'MeN(Ms)'], "[N](C)S(=O)(=O)C"),  # N-methyl-N-(methanesulfonyl (mesyl))amino
    _extra(['N(Me)Ns', 'NMeNs', 'N(Ns)Me', 'NsNMe', 'NsN(Me)', 'MeNNs', 'MeN(Ns)'], "[N](C)S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-methyl-N-(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['N(Me)OBn', 'N(OBn)Me', 'MeN(OBn)', 'BnON(Me)', 'BnONMe', 'MeNOBn'], "[N](C)OCc1ccccc1"),  # N-(benzyloxy)-N-methylamino
    _extra(['N(Me)Piv', 'NMePiv', 'N(Piv)Me', 'PivNMe', 'PivN(Me)', 'MeNPiv', 'MeN(Piv)'], "[N](C)C(=O)C(C)(C)C"),  # N-methyl-N-(pivaloyl)amino
    _extra(['N(Me)PMB', 'NMePMB', 'N(PMB)Me', 'PMBNMe', 'PMBN(Me)', 'MeNPMB', 'MeN(PMB)'], "[N](C)Cc1ccc(OC)cc1"),  # N-methyl-N-(4-methoxybenzyl)amino
    _extra(['N(Me)PMP', 'NMePMP', 'N(PMP)Me', 'PMPNMe', 'PMPN(Me)', 'MeNPMP', 'MeN(PMP)'], "[N](C)c1ccc(OC)cc1"),  # N-methyl-N-(4-methoxyphenyl)amino
    _extra(['N(Me)SEM', 'NMeSEM', 'N(SEM)Me', 'SEMNMe', 'SEMN(Me)', 'MeNSEM', 'MeN(SEM)'], "[N](C)COCC[Si](C)(C)C"),  # N-methyl-N-(2-(trimethylsilyl)ethoxymethyl)amino
    _extra(['N(Me)SES', 'NMeSES', 'N(SES)Me', 'SESNMe', 'SESN(Me)', 'MeNSES', 'MeN(SES)'], "[N](C)S(=O)(=O)CC[Si](C)(C)C"),  # N-methyl-N-(2-(trimethylsilyl)ethanesulfonyl)amino
    _extra(['N(Me)SO2CF3', 'NMeSO2CF3', 'MeNSO2CF3', 'N(SO2CF3)Me'], "[N](C)S(=O)(=O)C(F)(F)F"),  # N-methyl-N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['N(Me)SO2CH2CH2TMS', 'NMeSO2CH2CH2TMS', 'MeNSO2CH2CH2TMS', 'N(SO2CH2CH2TMS)Me'], "[N](C)S(=O)(=O)CC[Si](C)(C)C"),  # N-methyl-N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino
    _extra(['N(Me)SO2Me', 'NMeSO2Me', 'MeNSO2Me', 'N(SO2Me)Me'], "[N](C)S(=O)(=O)C"),  # N-methyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['N(Me)SO2p-Tol', 'NMeSO2p-Tol', 'MeNSO2p-Tol', 'N(SO2p-Tol)Me'], "[N](C)S(=O)(=O)c1ccc(C)cc1"),  # N-methyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(Me)SO2Ph', 'NMeSO2Ph', 'MeNSO2Ph', 'N(SO2Ph)Me'], "[N](C)S(=O)(=O)c1ccccc1"),  # N-methyl-N-(benzenesulfonyl (formula spelling))amino
    _extra(['N(Me)SO2pTol', 'NMeSO2pTol', 'MeNSO2pTol', 'N(SO2pTol)Me'], "[N](C)S(=O)(=O)c1ccc(C)cc1"),  # N-methyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(Me)SO2Tol', 'NMeSO2Tol', 'MeNSO2Tol', 'N(SO2Tol)Me'], "[N](C)S(=O)(=O)c1ccc(C)cc1"),  # N-methyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(Me)Teoc', 'NMeTeoc', 'N(Teoc)Me', 'TeocNMe', 'TeocN(Me)', 'MeNTeoc', 'MeN(Teoc)'], "[N](C)C(=O)OCC[Si](C)(C)C"),  # N-methyl-N-(2-(trimethylsilyl)ethoxycarbonyl)amino
    _extra(['N(Me)Tf', 'NMeTf', 'N(Tf)Me', 'TfNMe', 'TfN(Me)', 'MeNTf', 'MeN(Tf)'], "[N](C)S(=O)(=O)C(F)(F)F"),  # N-methyl-N-(trifluoromethanesulfonyl (triflyl))amino
    _extra(['N(Me)Tfa', 'NMeTfa', 'N(Tfa)Me', 'TfaNMe', 'TfaN(Me)', 'MeNTfa', 'MeN(Tfa)'], "[N](C)C(=O)C(F)(F)F"),  # N-methyl-N-(trifluoroacetyl)amino
    _extra(['N(Me)TMS', 'NMeTMS', 'N(TMS)Me', 'TMSNMe', 'TMSN(Me)', 'MeNTMS', 'MeN(TMS)'], "[N](C)[Si](C)(C)C"),  # N-methyl-N-(trimethylsilyl)amino
    _extra(['N(Me)Troc', 'NMeTroc', 'N(Troc)Me', 'TrocNMe', 'TrocN(Me)', 'MeNTroc', 'MeN(Troc)'], "[N](C)C(=O)OCC(Cl)(Cl)Cl"),  # N-methyl-N-(2,2,2-trichloroethoxycarbonyl)amino
    _extra(['N(Me)Trt', 'NMeTrt', 'N(Trt)Me', 'TrtNMe', 'TrtN(Me)', 'MeNTrt', 'MeN(Trt)'], "[N](C)C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # N-methyl-N-(triphenylmethyl (trityl))amino
    _extra(['N(Me)Ts', 'NMeTs', 'N(Ts)Me', 'TsNMe', 'TsN(Me)', 'MeNTs', 'MeN(Ts)'], "[N](C)S(=O)(=O)c1ccc(C)cc1"),  # N-methyl-N-(p-toluenesulfonyl (tosyl))amino
    _extra(['N(Me)Z', 'NMeZ', 'N(Z)Me', 'ZNMe', 'ZN(Me)', 'MeNZ', 'MeN(Z)'], "[N](C)C(=O)OCc1ccccc1"),  # N-methyl-N-(benzyloxycarbonyl)amino
    _extra(['N(Ms)2', 'NMs2', 'Ms2N', '(Ms)2N'], "[N](S(=O)(=O)C)S(=O)(=O)C"),  # bis(methanesulfonyl (mesyl))amino
    _extra(['N(Ms)NH2', 'H2NN(Ms)', 'N(NH2)Ms', 'MsN(NH2)'], "[N](N)S(=O)(=O)C"),  # 1-(Ms)hydrazin-1-yl
    _extra(['N(Ms)Ph', 'N(Ph)Ms', 'MsN(Ph)', 'PhN(Ms)', 'MsNPh', 'PhNMs'], "[N](c1ccccc1)S(C)(=O)=O"),  # N-mesyl-N-phenylamino
    _extra(['N(Ns)2', 'NNs2', 'Ns2N', '(Ns)2N'], "[N](S(=O)(=O)c1ccccc1[N+](=O)[O-])S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # bis(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['N(Ns)Allyl', 'N(Allyl)Ns', 'NsN(Allyl)', 'AllylN(Ns)', 'N(Ns)All', 'NsN(All)', 'NsNAll', 'AllNNs'], "[N](CC=C)S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-allyl-N-nosylamino
    _extra(['N(Ns)NH2', 'H2NN(Ns)', 'N(NH2)Ns', 'NsN(NH2)'], "[N](N)S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # 1-(Ns)hydrazin-1-yl
    _extra(['N(Ns)PMB', 'N(PMB)Ns', 'NsN(PMB)', 'PMBN(Ns)'], "[N](Cc1ccc(OC)cc1)S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(4-methoxybenzyl)-N-nosylamino
    _extra(['N(OBn)Boc', 'N(Boc)OBn', 'BocN(OBn)', 'BnON(Boc)', 'BnONBoc', 'BocNOBn'], "[N](OCc1ccccc1)C(=O)OC(C)(C)C"),  # N-(benzyloxy)-N-Boc-amino
    _extra(['N(OBn)Cbz', 'N(Cbz)OBn', 'CbzN(OBn)', 'BnON(Cbz)', 'BnONCbz'], "[N](OCc1ccccc1)C(=O)OCc1ccccc1"),  # N-(benzyloxy)-N-Cbz-amino
    _extra(['N(OBn)Ts', 'N(Ts)OBn', 'TsN(OBn)', 'BnON(Ts)'], "[N](OCc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-(benzyloxy)-N-tosylamino
    _extra(['N(OH)Ac', 'N(Ac)OH', 'AcN(OH)', 'HON(Ac)', 'HONAc', 'N(OH)C(O)Me'], "[N](O)C(C)=O"),  # N-acetyl-N-hydroxyamino (hydroxamate)
    _extra(['N(OH)Bn', 'N(Bn)OH', 'BnN(OH)', 'HON(Bn)', 'HONBn'], "[N](Cc1ccccc1)O"),  # N-benzyl-N-hydroxyamino
    _extra(['N(OH)Boc', 'N(Boc)OH', 'BocN(OH)', 'HON(Boc)', 'HONBoc'], "[N](O)C(=O)OC(C)(C)C"),  # N-Boc-N-hydroxyamino
    _extra(['N(OH)Me', 'N(Me)OH', 'MeN(OH)', 'HON(Me)', 'HONMe', 'MeNOH', 'N(OH)CH3', 'N(CH3)OH'], "[N](C)O"),  # N-hydroxy-N-methylamino
    _extra(['N(OMe)Ac', 'N(Ac)OMe', 'AcN(OMe)', 'MeON(Ac)', 'MeONAc'], "[N](OC)C(C)=O"),  # N-acetyl-N-methoxyamino
    _extra(['N(OMe)Bn', 'N(Bn)OMe', 'BnN(OMe)', 'MeON(Bn)', 'MeONBn', 'BnNOMe'], "[N](Cc1ccccc1)OC"),  # N-benzyl-N-methoxyamino
    _extra(['N(OMe)Boc', 'N(Boc)OMe', 'BocN(OMe)', 'MeON(Boc)', 'MeONBoc'], "[N](OC)C(=O)OC(C)(C)C"),  # N-Boc-N-methoxyamino
    _extra(['N(OMe)Me', 'N(Me)OMe', 'MeN(OMe)', 'MeON(Me)', 'MeONMe', 'MeNOMe', 'N(OMe)CH3', 'N(CH3)OMe'], "[N](C)OC"),  # N-methoxy-N-methylamino (Weinreb amide nitrogen)
    _extra(['N(OMe)Ts', 'N(Ts)OMe', 'TsN(OMe)', 'MeON(Ts)'], "[N](OC)S(=O)(=O)c1ccc(C)cc1"),  # N-methoxy-N-tosylamino
    _extra(['N(Piv)2', 'NPiv2', 'Piv2N', '(Piv)2N'], "[N](C(=O)C(C)(C)C)C(=O)C(C)(C)C"),  # bis(pivaloyl)amino
    _extra(['N(Piv)NH2', 'H2NN(Piv)', 'N(NH2)Piv', 'PivN(NH2)'], "[N](N)C(=O)C(C)(C)C"),  # 1-(Piv)hydrazin-1-yl
    _extra(['N(PMB)2', 'NPMB2', 'PMB2N', '(PMB)2N'], "[N](Cc1ccc(OC)cc1)Cc1ccc(OC)cc1"),  # bis(4-methoxybenzyl)amino
    _extra(['N(PMB)Allyl', 'N(Allyl)PMB', 'PMBN(Allyl)', 'AllylN(PMB)', 'N(PMB)All', 'PMBN(All)'], "[N](CC=C)Cc1ccc(OC)cc1"),  # N-allyl-N-(4-methoxybenzyl)amino
    _extra(['N(PMB)NH2', 'H2NN(PMB)', 'N(NH2)PMB', 'PMBN(NH2)'], "[N](N)Cc1ccc(OC)cc1"),  # 1-(PMB)hydrazin-1-yl
    _extra(['N(PMP)2', 'PMP2N', '(PMP)2N', 'NPMP2'], "[N](c1ccc(OC)cc1)c1ccc(OC)cc1"),  # bis(4-methoxyphenyl)amino
    _extra(['N(SEM)Boc', 'N(Boc)SEM', 'SEMN(Boc)', 'BocN(SEM)'], "[N](COCC[Si](C)(C)C)C(=O)OC(C)(C)C"),  # N-Boc-N-SEM-amino
    _extra(['N(SiMe3)2'], "[N]([Si](C)(C)C)[Si](C)(C)C"),  # bis(trimethylsilyl)amino
    _extra(['N(SO2CF3)2', '(SO2CF3)2N'], "[N](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"),  # bis(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['N(SO2CH2CH2TMS)2', '(SO2CH2CH2TMS)2N'], "[N](S(=O)(=O)CC[Si](C)(C)C)S(=O)(=O)CC[Si](C)(C)C"),  # bis(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino
    _extra(['N(SO2CH3)2', '(SO2CH3)2N'], "[N](S(=O)(=O)C)S(=O)(=O)C"),  # bis(methanesulfonyl (formula spelling))amino
    _extra(['N(SO2CH3)Me'], "[N](C)S(=O)(=O)C"),  # N-methyl-N-(methanesulfonyl (formula spelling))amino
    _extra(['N(SO2Me)2', '(SO2Me)2N'], "[N](S(=O)(=O)C)S(=O)(=O)C"),  # bis(methanesulfonyl (formula spelling))amino
    _extra(['N(SO2p-Tol)2', '(SO2p-Tol)2N'], "[N](S(=O)(=O)c1ccc(C)cc1)S(=O)(=O)c1ccc(C)cc1"),  # bis(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(SO2Ph)2', '(SO2Ph)2N'], "[N](S(=O)(=O)c1ccccc1)S(=O)(=O)c1ccccc1"),  # bis(benzenesulfonyl (formula spelling))amino
    _extra(['N(SO2pTol)2', '(SO2pTol)2N'], "[N](S(=O)(=O)c1ccc(C)cc1)S(=O)(=O)c1ccc(C)cc1"),  # bis(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(SO2Tol)2', '(SO2Tol)2N'], "[N](S(=O)(=O)c1ccc(C)cc1)S(=O)(=O)c1ccc(C)cc1"),  # bis(p-toluenesulfonyl (formula spelling))amino
    _extra(['N(TBS)2', 'NTBS2', 'TBS2N', '(TBS)2N'], "[N]([Si](C)(C)C(C)(C)C)[Si](C)(C)C(C)(C)C"),  # bis(tert-butyldimethylsilyl)amino
    _extra(['N(Tf)2', '(Tf)2N'], "[N](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"),  # bis(trifluoromethanesulfonyl (triflyl))amino
    _extra(['N(Tf)Ph', 'N(Ph)Tf', 'TfN(Ph)', 'PhN(Tf)', 'TfNPh', 'PhNTf'], "[N](c1ccccc1)S(=O)(=O)C(F)(F)F"),  # N-phenyl-N-triflylamino
    _extra(['N(Tfa)2', 'NTfa2', 'Tfa2N', '(Tfa)2N'], "[N](C(=O)C(F)(F)F)C(=O)C(F)(F)F"),  # bis(trifluoroacetyl)amino
    _extra(['N(Tfa)NH2', 'H2NN(Tfa)', 'N(NH2)Tfa', 'TfaN(NH2)'], "[N](N)C(=O)C(F)(F)F"),  # 1-(Tfa)hydrazin-1-yl
    _extra(['N(TMS)2', 'NTMS2', 'TMS2N', '(TMS)2N'], "[N]([Si](C)(C)C)[Si](C)(C)C"),  # bis(trimethylsilyl)amino
    _extra(['N(Tr)Me', 'N(Me)Tr', 'TrNMe', 'MeNTr'], "[N](C)C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # N-methyl-N-tritylamino
    _extra(['N(Troc)NH2', 'H2NN(Troc)', 'N(NH2)Troc', 'TrocN(NH2)'], "[N](N)C(=O)OCC(Cl)(Cl)Cl"),  # 1-(Troc)hydrazin-1-yl
    _extra(['N(Trt)2', 'NTrt2', 'Trt2N', '(Trt)2N'], "[N](C(c1ccccc1)(c1ccccc1)c1ccccc1)C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # bis(triphenylmethyl (trityl))amino
    _extra(['N(Trt)NH2', 'H2NN(Trt)', 'N(NH2)Trt', 'TrtN(NH2)'], "[N](N)C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # 1-(Trt)hydrazin-1-yl
    _extra(['N(Ts)2', 'NTs2', 'Ts2N', '(Ts)2N'], "[N](S(=O)(=O)c1ccc(C)cc1)S(=O)(=O)c1ccc(C)cc1"),  # bis(p-toluenesulfonyl (tosyl))amino
    _extra(['N(Ts)Ac', 'N(Ac)Ts', 'TsN(Ac)', 'AcN(Ts)', 'TsNAc', 'AcNTs'], "[N](C(C)=O)S(=O)(=O)c1ccc(C)cc1"),  # N-acetyl-N-tosylamino
    _extra(['N(Ts)Allyl', 'N(Allyl)Ts', 'TsN(Allyl)', 'AllylN(Ts)', 'N(Ts)All', 'TsN(All)', 'AllN(Ts)', 'TsNAll', 'AllNTs'], "[N](CC=C)S(=O)(=O)c1ccc(C)cc1"),  # N-allyl-N-tosylamino
    _extra(['N(Ts)Br', 'TsNBr', 'TsN(Br)', 'BrN(Ts)'], "[N](Br)S(=O)(=O)c1ccc(C)cc1"),  # N-bromo-N-tosylamino
    _extra(['N(Ts)Cl', 'N(Cl)Ts', 'TsN(Cl)', 'ClN(Ts)', 'TsNCl', 'ClNTs'], "[N](Cl)S(=O)(=O)c1ccc(C)cc1"),  # N-chloro-N-tosylamino (chloramine-T derived)
    _extra(['N(Ts)Et', 'N(Et)Ts', 'TsN(Et)', 'EtN(Ts)', 'TsNEt', 'EtNTs'], "[N](CC)S(=O)(=O)c1ccc(C)cc1"),  # N-ethyl-N-tosylamino
    _extra(['N(Ts)iPr', 'N(iPr)Ts', 'TsN(iPr)', 'iPrN(Ts)'], "[N](C(C)C)S(=O)(=O)c1ccc(C)cc1"),  # N-isopropyl-N-tosylamino
    _extra(['N(Ts)NH2', 'H2NN(Ts)', 'N(NH2)Ts', 'TsN(NH2)'], "[N](N)S(=O)(=O)c1ccc(C)cc1"),  # 1-(Ts)hydrazin-1-yl
    _extra(['N(Ts)Ph', 'N(Ph)Ts', 'TsN(Ph)', 'PhN(Ts)', 'TsNPh', 'PhNTs'], "[N](c1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-phenyl-N-tosylamino
    _extra(['N(Ts)PMB', 'N(PMB)Ts', 'TsN(PMB)', 'PMBN(Ts)', 'TsNPMB', 'PMBNTs'], "[N](Cc1ccc(OC)cc1)S(=O)(=O)c1ccc(C)cc1"),  # N-(4-methoxybenzyl)-N-tosylamino
    _extra(['N(Ts)tBu', 'N(tBu)Ts', 'TsN(tBu)', 'tBuN(Ts)'], "[N](C(C)(C)C)S(=O)(=O)c1ccc(C)cc1"),  # N-tert-butyl-N-tosylamino
    _extra(['N(Ts)TMS', 'N(TMS)Ts', 'TsN(TMS)', 'TMSN(Ts)'], "[N]([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1"),  # N-tosyl-N-(trimethylsilyl)amino
    _extra(['N(Z)2', 'NZ2', 'Z2N', '(Z)2N'], "[N](C(=O)OCc1ccccc1)C(=O)OCc1ccccc1"),  # bis(benzyloxycarbonyl)amino
    _extra(['N(Z)NH2', 'H2NN(Z)', 'N(NH2)Z', 'ZN(NH2)'], "[N](N)C(=O)OCc1ccccc1"),  # 1-(Z)hydrazin-1-yl
    _extra(['N2,4-DMB', '2,4-DMBN'], "[N]Cc1ccc(OC)cc1OC", 2),  # N-2,4-dimethoxybenzyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2,4-Dmb', '2,4-DmbN'], "[N]Cc1ccc(OC)cc1OC", 2),  # N-2,4-dimethoxybenzyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2,4-DNP', '2,4-DNPN'], "[N]c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", 2),  # N-2,4-dinitrophenyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2-Br-Z', '2-Br-ZN'], "[N]C(=O)OCc1ccccc1Br", 2),  # N-2-bromobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2-BrZ', '2-BrZN'], "[N]C(=O)OCc1ccccc1Br", 2),  # N-2-bromobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2-Cl-Z', '2-Cl-ZN'], "[N]C(=O)OCc1ccccc1Cl", 2),  # N-2-chlorobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2-ClZ', '2-ClZN'], "[N]C(=O)OCc1ccccc1Cl", 2),  # N-2-chlorobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2-Nap', '2-NapN'], "[N]Cc1ccc2ccccc2c1", 2),  # N-2-naphthylmethyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2-NAP', '2-NAPN'], "[N]Cc1ccc2ccccc2c1", 2),  # N-2-naphthylmethyl nitrogen with two bonds (ring/secondary N)
    _extra(['N2-Ns', '2-NsN'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (o-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['N3,4-DMB', '3,4-DMBN'], "[N]Cc1ccc(OC)c(OC)c1", 2),  # N-3,4-dimethoxybenzyl (veratryl) nitrogen with two bonds (ring/secondary N)
    _extra(['N3,4-Dmb', '3,4-DmbN'], "[N]Cc1ccc(OC)c(OC)c1", 2),  # N-3,4-dimethoxybenzyl (veratryl) nitrogen with two bonds (ring/secondary N)
    _extra(['N4-Ns', '4-NsN'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (p-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['N9-PhF', '9-PhFN'], "[N]C1(c2ccccc2)c2ccccc2-c2ccccc21", 2),  # N-9-phenylfluoren-9-yl nitrogen with two bonds (ring/secondary N)
    _extra(['NAc', 'AcN'], "[NH]C(=O)C"),  # N-(acetyl)amino
    _extra(['NAc', 'AcN'], "[N]C(=O)C", 2),  # N-acetyl nitrogen with two bonds (ring/secondary N)
    _extra(['NAdoc', 'AdocN'], "[N]C(=O)OC12CC3CC(CC(C3)C1)C2", 2),  # N-1-adamantyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NAll', 'AllN'], "[N]CC=C", 2),  # N-allyl nitrogen with two bonds (ring/secondary N)
    _extra(['NAlloc', 'AllocN'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NAllyl', 'AllylN'], "[N]CC=C", 2),  # N-allyl nitrogen with two bonds (ring/secondary N)
    _extra(['NAloc', 'AlocN'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NAn', 'AnN'], "[N]c1ccc(OC)cc1", 2),  # N-4-methoxyphenyl nitrogen with two bonds (ring/secondary N)
    _extra(['Nap', 'NAP', '2-Nap', '2-NAP'], "[CH2]c1ccc2ccccc2c1"),  # 2-naphthylmethyl
    _extra(['NBn', 'BnN'], "[NH]Cc1ccccc1"),  # N-(benzyl)amino
    _extra(['NBn', 'BnN'], "[N]Cc1ccccc1", 2),  # N-benzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBoc', 'BocN'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBOM', 'BOMN'], "[N]COCc1ccccc1", 2),  # N-benzyloxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBom', 'BomN'], "[N]COCc1ccccc1", 2),  # N-benzyloxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBpoc', 'BpocN'], "[N]C(=O)OC(C)(C)c1ccc(-c2ccccc2)cc1", 2),  # N-2-(4-biphenylyl)propan-2-yloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBrZ', 'BrZN'], "[N]C(=O)OCc1ccccc1Br", 2),  # N-2-bromobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBs', 'BsN'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (brosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NBsmoc', 'BsmocN'], "[N]C(=O)OCC1=Cc2ccccc2S1(=O)=O", 2),  # N-1,1-dioxobenzo[b]thiophen-2-ylmethoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBz', 'BzN'], "[N]C(=O)c1ccccc1", 2),  # N-benzoyl nitrogen with two bonds (ring/secondary N)
    _extra(['NBzh', 'BzhN'], "[N]C(c1ccccc1)c1ccccc1", 2),  # N-diphenylmethyl (benzhydryl) nitrogen with two bonds (ring/secondary N)
    _extra(['NBzl', 'BzlN'], "[N]Cc1ccccc1", 2),  # N-benzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NC(O)CF3'], "[N]C(=O)C(F)(F)F", 2),  # N-trifluoroacetyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)CH3'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)H'], "[N]C=O", 2),  # N-formyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)Me'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)OAllyl'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)OBn'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)OCH2CCl3'], "[N]C(=O)OCC(Cl)(Cl)Cl", 2),  # N-2,2,2-trichloroethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)OEt'], "[N]C(=O)OCC", 2),  # N-ethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)OMe'], "[N]C(=O)OC", 2),  # N-methoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)OtBu'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)Ph'], "[N]C(=O)c1ccccc1", 2),  # N-benzoyl (formula spelling) nitrogen with two bonds
    _extra(['NC(O)tBu'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl (formula spelling) nitrogen with two bonds
    _extra(['NC(Ph)3'], "[N]C(c1ccccc1)(c1ccccc1)c1ccccc1", 2),  # N-triphenylmethyl (trityl) nitrogen with two bonds (ring/secondary N)
    _extra(['NC6H4OMe'], "[N]c1ccc(OC)cc1", 2),  # N-4-methoxyphenyl nitrogen with two bonds (ring/secondary N)
    _extra(['NCbz', 'CbzN'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl)amino
    _extra(['NCbz', 'CbzN'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NCH(Ph)2'], "[N]C(c1ccccc1)c1ccccc1", 2),  # N-diphenylmethyl (benzhydryl) nitrogen with two bonds (ring/secondary N)
    _extra(['NCH2Ph'], "[N]Cc1ccccc1", 2),  # N-benzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NCHO'], "[NH]C=O"),  # N-(formyl (formula spelling))amino
    _extra(['NCHO'], "[N]C=O", 2),  # N-formyl (formula spelling) nitrogen with two bonds
    _extra(['NCHPh2'], "[N]C(c1ccccc1)c1ccccc1", 2),  # N-diphenylmethyl (benzhydryl) nitrogen with two bonds (ring/secondary N)
    _extra(['NClAc', 'ClAcN'], "[N]C(=O)CCl", 2),  # N-chloroacetyl nitrogen with two bonds (ring/secondary N)
    _extra(['NClZ', 'ClZN'], "[N]C(=O)OCc1ccccc1Cl", 2),  # N-2-chlorobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NCMe2Ph'], "[N]C(C)(C)c1ccccc1", 2),  # N-cumyl (2-phenylpropan-2-yl) nitrogen with two bonds (ring/secondary N)
    _extra(['NCO2All'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2Allyl'], "[N]C(=O)OCC=C", 2),  # N-allyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2Bn'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2But'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2CH2CCl3'], "[N]C(=O)OCC(Cl)(Cl)Cl", 2),  # N-2,2,2-trichloroethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2CH2Ph'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2Et'], "[N]C(=O)OCC", 2),  # N-ethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2Me'], "[N]C(=O)OC", 2),  # N-methoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2t-Bu'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCO2tBu'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCoc', 'CocN'], "[N]C(=O)OCC=Cc1ccccc1", 2),  # N-cinnamyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NCOCF3'], "[N]C(=O)C(F)(F)F", 2),  # N-trifluoroacetyl (formula spelling) nitrogen with two bonds
    _extra(['NCOCH2Cl'], "[N]C(=O)CCl", 2),  # N-chloroacetyl (formula spelling) nitrogen with two bonds
    _extra(['NCOCH3'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['NCOMe'], "[N]C(=O)C", 2),  # N-acetyl (formula spelling) nitrogen with two bonds
    _extra(['NCOOBn'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCOOEt'], "[N]C(=O)OCC", 2),  # N-ethoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCOOMe'], "[N]C(=O)OC", 2),  # N-methoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCOOtBu'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['NCOPh'], "[N]C(=O)c1ccccc1", 2),  # N-benzoyl (formula spelling) nitrogen with two bonds
    _extra(['NCOt-Bu'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl (formula spelling) nitrogen with two bonds
    _extra(['NCOtBu'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl (formula spelling) nitrogen with two bonds
    _extra(['NCPh3'], "[N]C(c1ccccc1)(c1ccccc1)c1ccccc1", 2),  # N-triphenylmethyl (trityl) nitrogen with two bonds (ring/secondary N)
    _extra(['NCPhMe2'], "[N]C(C)(C)c1ccccc1", 2),  # N-cumyl (2-phenylpropan-2-yl) nitrogen with two bonds (ring/secondary N)
    _extra(['NCum', 'CumN'], "[N]C(C)(C)c1ccccc1", 2),  # N-cumyl (2-phenylpropan-2-yl) nitrogen with two bonds (ring/secondary N)
    _extra(['NDde', 'DdeN'], "[N]C(C)=C1C(=O)CC(C)(C)CC1=O", 2),  # N-1-(4,4-dimethyl-2,6-dioxocyclohexylidene)ethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDdz', 'DdzN'], "[N]C(=O)OC(C)(C)c1cc(OC)cc(OC)c1", 2),  # N-2-(3,5-dimethoxyphenyl)propan-2-yloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDMB', 'DMBN'], "[N]Cc1ccc(OC)cc1OC", 2),  # N-2,4-dimethoxybenzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDmb', 'DmbN'], "[N]Cc1ccc(OC)cc1OC", 2),  # N-2,4-dimethoxybenzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDMT', 'DMTN'], "[N]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1", 2),  # N-4,4'-dimethoxytrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NDmt', 'DmtN'], "[N]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1", 2),  # N-4,4'-dimethoxytrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NDMTr', 'DMTrN'], "[N]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1", 2),  # N-4,4'-dimethoxytrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NDNBS', 'DNBSN'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", 2),  # N-2,4-dinitrobenzenesulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDnp', 'DnpN'], "[N]c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", 2),  # N-2,4-dinitrophenyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDNP', 'DNPN'], "[N]c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", 2),  # N-2,4-dinitrophenyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDNs', 'DNsN'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", 2),  # N-2,4-dinitrobenzenesulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NdNs', 'dNsN'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", 2),  # N-2,4-dinitrobenzenesulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDns', 'DnsN'], "[N]S(=O)(=O)c1cccc2c(N(C)C)cccc12", 2),  # N-5-(dimethylamino)naphthalene-1-sulfonyl (dansyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NDoc', 'DocN'], "[N]C(=O)OC(C(C)C)C(C)C", 2),  # N-2,4-dimethylpent-3-yloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NDpm', 'DpmN'], "[N]C(c1ccccc1)c1ccccc1", 2),  # N-diphenylmethyl (benzhydryl) nitrogen with two bonds (ring/secondary N)
    _extra(['NEoc', 'EocN'], "[N]C(=O)OCC", 2),  # N-ethoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NEt3+', 'Et3N+', 'N+Et3', 'N(Et)3+', 'NEt3', 'Et3N'], "[N+](CC)(CC)CC"),  # triethylammonio
    _extra(['Nf'], "[S](=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),  # nonafluorobutanesulfonyl (nonaflyl)
    _extra(['NFmoc', 'FmocN'], "[NH]C(=O)OCC1c2ccccc2-c2ccccc21"),  # N-(9-fluorenylmethoxycarbonyl)amino
    _extra(['NFmoc', 'FmocN'], "[N]C(=O)OCC1c2ccccc2-c2ccccc21", 2),  # N-9-fluorenylmethoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NFor', 'ForN'], "[N]C=O", 2),  # N-formyl nitrogen with two bonds (ring/secondary N)
    _extra(['NH2,4-DMB', '2,4-DMBHN', '2,4-DMBNH', 'N2,4-DMB', '2,4-DMBN'], "[NH]Cc1ccc(OC)cc1OC"),  # N-(2,4-dimethoxybenzyl)amino
    _extra(['NH2,4-Dmb', '2,4-DmbHN', '2,4-DmbNH', 'N2,4-Dmb', '2,4-DmbN'], "[NH]Cc1ccc(OC)cc1OC"),  # N-(2,4-dimethoxybenzyl)amino
    _extra(['NH2,4-DNP', '2,4-DNPHN', '2,4-DNPNH', 'N2,4-DNP', '2,4-DNPN'], "[NH]c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # N-(2,4-dinitrophenyl)amino
    _extra(['NH2-Br-Z', '2-Br-ZHN', '2-Br-ZNH', 'N2-Br-Z', '2-Br-ZN'], "[NH]C(=O)OCc1ccccc1Br"),  # N-(2-bromobenzyloxycarbonyl)amino
    _extra(['NH2-BrZ', '2-BrZHN', '2-BrZNH', 'N2-BrZ', '2-BrZN'], "[NH]C(=O)OCc1ccccc1Br"),  # N-(2-bromobenzyloxycarbonyl)amino
    _extra(['NH2-Cl-Z', '2-Cl-ZHN', '2-Cl-ZNH', 'N2-Cl-Z', '2-Cl-ZN'], "[NH]C(=O)OCc1ccccc1Cl"),  # N-(2-chlorobenzyloxycarbonyl)amino
    _extra(['NH2-ClZ', '2-ClZHN', '2-ClZNH', 'N2-ClZ', '2-ClZN'], "[NH]C(=O)OCc1ccccc1Cl"),  # N-(2-chlorobenzyloxycarbonyl)amino
    _extra(['NH2-Nap', '2-NapHN', '2-NapNH', 'N2-Nap', '2-NapN'], "[NH]Cc1ccc2ccccc2c1"),  # N-(2-naphthylmethyl)amino
    _extra(['NH2-NAP', '2-NAPHN', '2-NAPNH', 'N2-NAP', '2-NAPN'], "[NH]Cc1ccc2ccccc2c1"),  # N-(2-naphthylmethyl)amino
    _extra(['NH2-Ns', '2-NsHN', '2-NsNH', 'N2-Ns', '2-NsN'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['NH2.HCl', 'HCl.H2N', 'H2N.HCl'], "[NH3+].[Cl-]"),  # amine hydrochloride
    _extra(['NH3,4-DMB', '3,4-DMBHN', '3,4-DMBNH', 'N3,4-DMB', '3,4-DMBN'], "[NH]Cc1ccc(OC)c(OC)c1"),  # N-(3,4-dimethoxybenzyl (veratryl))amino
    _extra(['NH3,4-Dmb', '3,4-DmbHN', '3,4-DmbNH', 'N3,4-Dmb', '3,4-DmbN'], "[NH]Cc1ccc(OC)c(OC)c1"),  # N-(3,4-dimethoxybenzyl (veratryl))amino
    _extra(['NH3Br', 'NH3+Br-', 'BrH3N'], "[NH3+].[Br-]"),  # ammonium bromide (amine hydrobromide)
    _extra(['NH3Cl', 'NH3+Cl-', 'ClH3N', 'Cl-H3N+', 'H3N+Cl-'], "[NH3+].[Cl-]"),  # ammonium chloride (amine hydrochloride)
    _extra(['NH3TFA', 'NH3+TFA-', 'NH3+CF3CO2-', 'NH3+CF3COO-'], "[NH3+].[O-]C(=O)C(F)(F)F"),  # ammonium trifluoroacetate
    _extra(['NH4-Ns', '4-NsHN', '4-NsNH', 'N4-Ns', '4-NsN'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (p-nosyl))amino
    _extra(['NH9-PhF', '9-PhFHN', '9-PhFNH', 'N9-PhF', '9-PhFN'], "[NH]C1(c2ccccc2)c2ccccc2-c2ccccc21"),  # N-(9-phenylfluoren-9-yl)amino
    _extra(['NHAdoc', 'AdocHN', 'AdocNH', 'NAdoc', 'AdocN'], "[NH]C(=O)OC12CC3CC(CC(C3)C1)C2"),  # N-(1-adamantyloxycarbonyl)amino
    _extra(['NHAll', 'AllHN', 'AllNH', 'NAll', 'AllN'], "[NH]CC=C"),  # N-(allyl)amino
    _extra(['NHAlloc', 'AllocHN', 'AllocNH', 'NAlloc', 'AllocN'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl)amino
    _extra(['NHAllyl', 'AllylHN', 'AllylNH', 'NAllyl', 'AllylN'], "[NH]CC=C"),  # N-(allyl)amino
    _extra(['NHAloc', 'AlocHN', 'AlocNH', 'NAloc', 'AlocN'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl)amino
    _extra(['NHAn', 'AnHN', 'AnNH', 'NAn', 'AnN'], "[NH]c1ccc(OC)cc1"),  # N-(4-methoxyphenyl)amino
    _extra(['NHBOM', 'BOMHN', 'BOMNH', 'NBOM', 'BOMN'], "[NH]COCc1ccccc1"),  # N-(benzyloxymethyl)amino
    _extra(['NHBom', 'BomHN', 'BomNH', 'NBom', 'BomN'], "[NH]COCc1ccccc1"),  # N-(benzyloxymethyl)amino
    _extra(['NHBpoc', 'BpocHN', 'BpocNH', 'NBpoc', 'BpocN'], "[NH]C(=O)OC(C)(C)c1ccc(-c2ccccc2)cc1"),  # N-(2-(4-biphenylyl)propan-2-yloxycarbonyl)amino
    _extra(['NHBrZ', 'BrZHN', 'BrZNH', 'NBrZ', 'BrZN'], "[NH]C(=O)OCc1ccccc1Br"),  # N-(2-bromobenzyloxycarbonyl)amino
    _extra(['NHBs', 'BsHN', 'BsNH', 'NBs', 'BsN'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (brosyl))amino
    _extra(['NHBsmoc', 'BsmocHN', 'BsmocNH', 'NBsmoc', 'BsmocN'], "[NH]C(=O)OCC1=Cc2ccccc2S1(=O)=O"),  # N-(1,1-dioxobenzo[b]thiophen-2-ylmethoxycarbonyl)amino
    _extra(['NHBz', 'BzHN', 'BzNH', 'NBz', 'BzN'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl)amino
    _extra(['NHBzh', 'BzhHN', 'BzhNH', 'NBzh', 'BzhN'], "[NH]C(c1ccccc1)c1ccccc1"),  # N-(diphenylmethyl (benzhydryl))amino
    _extra(['NHBzl', 'BzlHN', 'BzlNH', 'NBzl', 'BzlN'], "[NH]Cc1ccccc1"),  # N-(benzyl)amino
    _extra(['NHC(O)CF3', 'NC(O)CF3'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino
    _extra(['NHC(O)CH3', 'NC(O)CH3'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino
    _extra(['NHC(O)H', 'NC(O)H'], "[NH]C=O"),  # N-(formyl (formula spelling))amino
    _extra(['NHC(O)Me', 'NC(O)Me'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino
    _extra(['NHC(O)OAllyl', 'NC(O)OAllyl'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino
    _extra(['NHC(O)OBn', 'NC(O)OBn'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['NHC(O)OCH2CCl3', 'NC(O)OCH2CCl3'], "[NH]C(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino
    _extra(['NHC(O)OEt', 'NC(O)OEt'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino
    _extra(['NHC(O)OMe', 'NC(O)OMe'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino
    _extra(['NHC(O)OtBu', 'NC(O)OtBu'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['NHC(O)Ph', 'NC(O)Ph'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino
    _extra(['NHC(O)tBu', 'NC(O)tBu'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino
    _extra(['NHC(Ph)3', 'NC(Ph)3'], "[NH]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # N-(triphenylmethyl (trityl))amino
    _extra(['NHC6H4OMe', 'NC6H4OMe'], "[NH]c1ccc(OC)cc1"),  # N-(4-methoxyphenyl)amino
    _extra(['NHCH(Ph)2', 'NCH(Ph)2'], "[NH]C(c1ccccc1)c1ccccc1"),  # N-(diphenylmethyl (benzhydryl))amino
    _extra(['NHCH2Ph', 'NCH2Ph'], "[NH]Cc1ccccc1"),  # N-(benzyl)amino
    _extra(['NHCHPh2', 'NCHPh2'], "[NH]C(c1ccccc1)c1ccccc1"),  # N-(diphenylmethyl (benzhydryl))amino
    _extra(['NHClAc', 'ClAcHN', 'ClAcNH', 'NClAc', 'ClAcN'], "[NH]C(=O)CCl"),  # N-(chloroacetyl)amino
    _extra(['NHClZ', 'ClZHN', 'ClZNH', 'NClZ', 'ClZN'], "[NH]C(=O)OCc1ccccc1Cl"),  # N-(2-chlorobenzyloxycarbonyl)amino
    _extra(['NHCMe2Ph', 'NCMe2Ph'], "[NH]C(C)(C)c1ccccc1"),  # N-(cumyl (2-phenylpropan-2-yl))amino
    _extra(['NHCO2All', 'NCO2All'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino
    _extra(['NHCO2Allyl', 'NCO2Allyl'], "[NH]C(=O)OCC=C"),  # N-(allyloxycarbonyl (formula spelling))amino
    _extra(['NHCO2Bn', 'NCO2Bn'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['NHCO2But', 'NCO2But'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['NHCO2CH2CCl3', 'NCO2CH2CCl3'], "[NH]C(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxycarbonyl (formula spelling))amino
    _extra(['NHCO2CH2Ph', 'NCO2CH2Ph'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['NHCO2Et', 'NCO2Et'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino
    _extra(['NHCO2Me', 'NCO2Me'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino
    _extra(['NHCO2t-Bu', 'NCO2t-Bu'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['NHCO2tBu', 'NCO2tBu'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['NHCoc', 'CocHN', 'CocNH', 'NCoc', 'CocN'], "[NH]C(=O)OCC=Cc1ccccc1"),  # N-(cinnamyloxycarbonyl)amino
    _extra(['NHCOCF3', 'NCOCF3'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl (formula spelling))amino
    _extra(['NHCOCH2Cl', 'NCOCH2Cl'], "[NH]C(=O)CCl"),  # N-(chloroacetyl (formula spelling))amino
    _extra(['NHCOCH3', 'NCOCH3'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino
    _extra(['NHCOMe', 'NCOMe'], "[NH]C(=O)C"),  # N-(acetyl (formula spelling))amino
    _extra(['NHCOOBn', 'NCOOBn'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino
    _extra(['NHCOOEt', 'NCOOEt'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl (formula spelling))amino
    _extra(['NHCOOMe', 'NCOOMe'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl (formula spelling))amino
    _extra(['NHCOOtBu', 'NCOOtBu'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['NHCOPh', 'NCOPh'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino
    _extra(['NHCOt-Bu', 'NCOt-Bu'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino
    _extra(['NHCOtBu', 'NCOtBu'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino
    _extra(['NHCPh3', 'NCPh3'], "[NH]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # N-(triphenylmethyl (trityl))amino
    _extra(['NHCPhMe2', 'NCPhMe2'], "[NH]C(C)(C)c1ccccc1"),  # N-(cumyl (2-phenylpropan-2-yl))amino
    _extra(['NHCum', 'CumHN', 'CumNH', 'NCum', 'CumN'], "[NH]C(C)(C)c1ccccc1"),  # N-(cumyl (2-phenylpropan-2-yl))amino
    _extra(['NHDde', 'DdeHN', 'DdeNH', 'NDde', 'DdeN'], "[NH]C(C)=C1C(=O)CC(C)(C)CC1=O"),  # N-(1-(4,4-dimethyl-2,6-dioxocyclohexylidene)ethyl)amino
    _extra(['NHDdz', 'DdzHN', 'DdzNH', 'NDdz', 'DdzN'], "[NH]C(=O)OC(C)(C)c1cc(OC)cc(OC)c1"),  # N-(2-(3,5-dimethoxyphenyl)propan-2-yloxycarbonyl)amino
    _extra(['NHDMB', 'DMBHN', 'DMBNH', 'NDMB', 'DMBN'], "[NH]Cc1ccc(OC)cc1OC"),  # N-(2,4-dimethoxybenzyl)amino
    _extra(['NHDmb', 'DmbHN', 'DmbNH', 'NDmb', 'DmbN'], "[NH]Cc1ccc(OC)cc1OC"),  # N-(2,4-dimethoxybenzyl)amino
    _extra(['NHDMT', 'DMTHN', 'DMTNH', 'NDMT', 'DMTN'], "[NH]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1"),  # N-(4,4'-dimethoxytrityl)amino
    _extra(['NHDmt', 'DmtHN', 'DmtNH', 'NDmt', 'DmtN'], "[NH]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1"),  # N-(4,4'-dimethoxytrityl)amino
    _extra(['NHDMTr', 'DMTrHN', 'DMTrNH', 'NDMTr', 'DMTrN'], "[NH]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1"),  # N-(4,4'-dimethoxytrityl)amino
    _extra(['NHDNBS', 'DNBSHN', 'DNBSNH', 'NDNBS', 'DNBSN'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # N-(2,4-dinitrobenzenesulfonyl)amino
    _extra(['NHDnp', 'DnpHN', 'DnpNH', 'NDnp', 'DnpN'], "[NH]c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # N-(2,4-dinitrophenyl)amino
    _extra(['NHDNP', 'DNPHN', 'DNPNH', 'NDNP', 'DNPN'], "[NH]c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # N-(2,4-dinitrophenyl)amino
    _extra(['NHDNs', 'DNsHN', 'DNsNH', 'NDNs', 'DNsN'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # N-(2,4-dinitrobenzenesulfonyl)amino
    _extra(['NHdNs', 'dNsHN', 'dNsNH', 'NdNs', 'dNsN'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # N-(2,4-dinitrobenzenesulfonyl)amino
    _extra(['NHDns', 'DnsHN', 'DnsNH', 'NDns', 'DnsN'], "[NH]S(=O)(=O)c1cccc2c(N(C)C)cccc12"),  # N-(5-(dimethylamino)naphthalene-1-sulfonyl (dansyl))amino
    _extra(['NHDoc', 'DocHN', 'DocNH', 'NDoc', 'DocN'], "[NH]C(=O)OC(C(C)C)C(C)C"),  # N-(2,4-dimethylpent-3-yloxycarbonyl)amino
    _extra(['NHDpm', 'DpmHN', 'DpmNH', 'NDpm', 'DpmN'], "[NH]C(c1ccccc1)c1ccccc1"),  # N-(diphenylmethyl (benzhydryl))amino
    _extra(['NHEoc', 'EocHN', 'EocNH', 'NEoc', 'EocN'], "[NH]C(=O)OCC"),  # N-(ethoxycarbonyl)amino
    _extra(['NHFor', 'ForHN', 'ForNH', 'NFor', 'ForN'], "[NH]C=O"),  # N-(formyl)amino
    _extra(['NHiv-Dde', 'iv-DdeHN', 'iv-DdeNH', 'Niv-Dde', 'iv-DdeN'], "[NH]C(CC(C)C)=C1C(=O)CC(C)(C)CC1=O"),  # N-(1-(4,4-dimethyl-2,6-dioxocyclohexylidene)-3-methylbutyl)amino
    _extra(['NHivDde', 'ivDdeHN', 'ivDdeNH', 'NivDde', 'ivDdeN'], "[NH]C(CC(C)C)=C1C(=O)CC(C)(C)CC1=O"),  # N-(1-(4,4-dimethyl-2,6-dioxocyclohexylidene)-3-methylbutyl)amino
    _extra(['NHMbh', 'MbhHN', 'MbhNH', 'NMbh', 'MbhN'], "[NH]C(c1ccc(OC)cc1)c1ccc(OC)cc1"),  # N-(4,4'-dimethoxybenzhydryl)amino
    _extra(['NHMEM', 'MEMHN', 'MEMNH', 'NMEM', 'MEMN'], "[NH]COCCOC"),  # N-(2-methoxyethoxymethyl)amino
    _extra(['NHMeOZ', 'MeOZHN', 'MeOZNH', 'NMeOZ', 'MeOZN'], "[NH]C(=O)OCc1ccc(OC)cc1"),  # N-(4-methoxybenzyloxycarbonyl)amino
    _extra(['NHMes(O)S', 'Mes(O)SHN', 'Mes(O)SNH', 'NMes(O)S', 'Mes(O)SN'], "[NH]S(=O)c1c(C)cc(C)cc1C"),  # N-(2,4,6-trimethylbenzenesulfinyl (mesitylsulfinyl))amino
    _extra(['NHMesS(O)', 'MesS(O)HN', 'MesS(O)NH', 'NMesS(O)', 'MesS(O)N'], "[NH]S(=O)c1c(C)cc(C)cc1C"),  # N-(2,4,6-trimethylbenzenesulfinyl (mesitylsulfinyl))amino
    _extra(['NHMesSO', 'MesSOHN', 'MesSONH', 'NMesSO', 'MesSON'], "[NH]S(=O)c1c(C)cc(C)cc1C"),  # N-(2,4,6-trimethylbenzenesulfinyl (mesitylsulfinyl))amino
    _extra(['NHMmt', 'MmtHN', 'MmtNH', 'NMmt', 'MmtN'], "[NH]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1"),  # N-(4-methoxytrityl)amino
    _extra(['NHMMT', 'MMTHN', 'MMTNH', 'NMMT', 'MMTN'], "[NH]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1"),  # N-(4-methoxytrityl)amino
    _extra(['NHMMTr', 'MMTrHN', 'MMTrNH', 'NMMTr', 'MMTrN'], "[NH]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1"),  # N-(4-methoxytrityl)amino
    _extra(['NHMmtr', 'MmtrHN', 'MmtrNH', 'NMmtr', 'MmtrN'], "[NH]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1"),  # N-(4-methoxytrityl)amino
    _extra(['NHMoc', 'MocHN', 'MocNH', 'NMoc', 'MocN'], "[NH]C(=O)OC"),  # N-(methoxycarbonyl)amino
    _extra(['NHMOM', 'MOMHN', 'MOMNH', 'NMOM', 'MOMN'], "[NH]COC"),  # N-(methoxymethyl)amino
    _extra(['NHMoz', 'MozHN', 'MozNH', 'NMoz', 'MozN'], "[NH]C(=O)OCc1ccc(OC)cc1"),  # N-(4-methoxybenzyloxycarbonyl)amino
    _extra(['NHMPM', 'MPMHN', 'MPMNH', 'NMPM', 'MPMN'], "[NH]Cc1ccc(OC)cc1"),  # N-(4-methoxybenzyl)amino
    _extra(['NHMsc', 'MscHN', 'MscNH', 'NMsc', 'MscN'], "[NH]C(=O)OCCS(C)(=O)=O"),  # N-(2-(methylsulfonyl)ethoxycarbonyl)amino
    _extra(['NHMtr', 'MtrHN', 'MtrNH', 'NMtr', 'MtrN'], "[NH]S(=O)(=O)c1c(C)cc(OC)c(C)c1C"),  # N-(4-methoxy-2,3,6-trimethylbenzenesulfonyl)amino
    _extra(['NHMts', 'MtsHN', 'MtsNH', 'NMts', 'MtsN'], "[NH]S(=O)(=O)c1c(C)cc(C)cc1C"),  # N-(mesitylenesulfonyl (2,4,6-trimethylbenzenesulfonyl))amino
    _extra(['NHMtt', 'MttHN', 'MttNH', 'NMtt', 'MttN'], "[NH]C(c1ccc(C)cc1)(c1ccccc1)c1ccccc1"),  # N-(4-methyltrityl)amino
    _extra(['NHMttr', 'MttrHN', 'MttrNH', 'NMttr', 'MttrN'], "[NH]C(c1ccc(C)cc1)(c1ccccc1)c1ccccc1"),  # N-(4-methyltrityl)amino
    _extra(['NHN(Boc)2', 'Boc2NNH', '(Boc)2NNH', 'NHNBoc2'], "[NH]N(C(=O)OC(C)(C)C)C(=O)OC(C)(C)C"),  # 2,2-bis(tert-butoxycarbonyl)hydrazinyl
    _extra(['NHNap', 'NapHN', 'NapNH', 'NNap', 'NapN'], "[NH]Cc1ccc2ccccc2c1"),  # N-(2-naphthylmethyl)amino
    _extra(['NHNAP', 'NAPHN', 'NAPNH', 'NNAP', 'NAPN'], "[NH]Cc1ccc2ccccc2c1"),  # N-(2-naphthylmethyl)amino
    _extra(['NHNbs', 'NbsHN', 'NbsNH', 'NNbs', 'NbsN'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (p-nosyl))amino
    _extra(['NHNf', 'NfHN', 'NfNH', 'NNf', 'NfN'], "[NH]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),  # N-(nonafluorobutanesulfonyl (nonaflyl))amino
    _extra(['NHNHAc', 'AcHNHN', 'AcNHNH', 'AcHNNH'], "[NH]NC(=O)C"),  # 2-(Ac)hydrazinyl
    _extra(['NHNHAlloc', 'AllocHNHN', 'AllocNHNH', 'AllocHNNH'], "[NH]NC(=O)OCC=C"),  # 2-(Alloc)hydrazinyl
    _extra(['NHNHBn', 'BnHNHN', 'BnNHNH', 'BnHNNH'], "[NH]NCc1ccccc1"),  # 2-(Bn)hydrazinyl
    _extra(['NHNHBoc', 'BocHNHN', 'BocNHNH', 'BocHNNH'], "[NH]NC(=O)OC(C)(C)C"),  # 2-(Boc)hydrazinyl
    _extra(['NHNHBz', 'BzHNHN', 'BzNHNH', 'BzHNNH'], "[NH]NC(=O)c1ccccc1"),  # 2-(Bz)hydrazinyl
    _extra(['NHNHCbz', 'CbzHNHN', 'CbzNHNH', 'CbzHNNH'], "[NH]NC(=O)OCc1ccccc1"),  # 2-(Cbz)hydrazinyl
    _extra(['NHNHFmoc', 'FmocHNHN', 'FmocNHNH', 'FmocHNNH'], "[NH]NC(=O)OCC1c2ccccc2-c2ccccc21"),  # 2-(Fmoc)hydrazinyl
    _extra(['NHNHMs', 'MsHNHN', 'MsNHNH', 'MsHNNH'], "[NH]NS(=O)(=O)C"),  # 2-(Ms)hydrazinyl
    _extra(['NHNHNs', 'NsHNHN', 'NsNHNH', 'NsHNNH'], "[NH]NS(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # 2-(Ns)hydrazinyl
    _extra(['NHNHPiv', 'PivHNHN', 'PivNHNH', 'PivHNNH'], "[NH]NC(=O)C(C)(C)C"),  # 2-(Piv)hydrazinyl
    _extra(['NHNHPMB', 'PMBHNHN', 'PMBNHNH', 'PMBHNNH'], "[NH]NCc1ccc(OC)cc1"),  # 2-(PMB)hydrazinyl
    _extra(['NHNHTfa', 'TfaHNHN', 'TfaNHNH', 'TfaHNNH'], "[NH]NC(=O)C(F)(F)F"),  # 2-(Tfa)hydrazinyl
    _extra(['NHNHTroc', 'TrocHNHN', 'TrocNHNH', 'TrocHNNH'], "[NH]NC(=O)OCC(Cl)(Cl)Cl"),  # 2-(Troc)hydrazinyl
    _extra(['NHNHTrt', 'TrtHNHN', 'TrtNHNH', 'TrtHNNH'], "[NH]NC(c1ccccc1)(c1ccccc1)c1ccccc1"),  # 2-(Trt)hydrazinyl
    _extra(['NHNHTs', 'TsHNHN', 'TsNHNH', 'TsHNNH'], "[NH]NS(=O)(=O)c1ccc(C)cc1"),  # 2-(Ts)hydrazinyl
    _extra(['NHNHZ', 'ZHNHN', 'ZNHNH', 'ZHNNH'], "[NH]NC(=O)OCc1ccccc1"),  # 2-(Z)hydrazinyl
    _extra(['NHNoc', 'NocHN', 'NocNH', 'NNoc', 'NocN'], "[NH]C(=O)OCC=Cc1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrocinnamyloxycarbonyl)amino
    _extra(['NHNos', 'NosHN', 'NosNH', 'NNos', 'NosN'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['NHNps', 'NpsHN', 'NpsNH', 'NNps', 'NpsN'], "[NH]Sc1ccccc1[N+](=O)[O-]"),  # N-(2-nitrophenylsulfenyl)amino
    _extra(['NHNs', 'NsHN', 'NsNH', 'NNs', 'NsN'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['NHNsc', 'NscHN', 'NscNH', 'NNsc', 'NscN'], "[NH]C(=O)OCCS(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(2-(4-nitrophenylsulfonyl)ethoxycarbonyl)amino
    _extra(['NHNvoc', 'NvocHN', 'NvocNH', 'NNvoc', 'NvocN'], "[NH]C(=O)OCc1cc(OC)c(OC)cc1[N+](=O)[O-]"),  # N-(6-nitroveratryloxycarbonyl (4,5-dimethoxy-2-nitrobenzyloxycarbonyl))amino
    _extra(['NHNVoc', 'NVocHN', 'NVocNH', 'NNVoc', 'NVocN'], "[NH]C(=O)OCc1cc(OC)c(OC)cc1[N+](=O)[O-]"),  # N-(6-nitroveratryloxycarbonyl (4,5-dimethoxy-2-nitrobenzyloxycarbonyl))amino
    _extra(['NHo-Ns', 'o-NsHN', 'o-NsNH', 'No-Ns', 'o-NsN'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['NHOAc', 'AcONH', 'AcOHN', 'NHOC(O)Me', 'NHOCOCH3'], "[NH]OC(C)=O"),  # acetoxyamino
    _extra(['NHOAll', 'AllONH', 'NHOAllyl', 'AllylONH'], "[NH]OCC=C"),  # (allyloxy)amino
    _extra(['NHOBn', 'BnONH', 'BnOHN', 'NHOBzl', 'BzlONH', 'NHOCH2Ph', 'PhCH2ONH'], "[NH]OCc1ccccc1"),  # (benzyloxy)amino
    _extra(['NHOBz', 'BzONH', 'BzOHN'], "[NH]OC(=O)c1ccccc1"),  # (benzoyloxy)amino
    _extra(['NHOMe', 'MeONH', 'MeOHN', 'NHOCH3', 'H3CONH'], "[NH]OC"),  # methoxyamino
    _extra(['NHOMOM', 'MOMONH'], "[NH]OCOC"),  # (methoxymethoxy)amino
    _extra(['NHoNs', 'oNsHN', 'oNsNH', 'NoNs', 'oNsN'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (o-nosyl))amino
    _extra(['NHOPiv', 'PivONH', 'PivOHN'], "[NH]OC(=O)C(C)(C)C"),  # (pivaloyloxy)amino
    _extra(['NHOPMB', 'PMBONH', 'PMBOHN'], "[NH]OCc1ccc(OC)cc1"),  # (4-methoxybenzyloxy)amino
    _extra(['NHOTBS', 'TBSONH', 'TBSOHN', 'NHOTBDMS', 'TBDMSONH', 'NHOSiMe2tBu'], "[NH]O[Si](C)(C)C(C)(C)C"),  # (tert-butyldimethylsilyloxy)amino
    _extra(['NHOtBu', 'tBuONH', 'tBuOHN', 'NHOt-Bu', 't-BuONH', 'NHOBut'], "[NH]OC(C)(C)C"),  # (tert-butoxy)amino
    _extra(['NHOTHP', 'THPONH', 'THPOHN'], "[NH]OC1CCCCO1"),  # (tetrahydropyran-2-yloxy)amino
    _extra(['NHOTMS', 'TMSONH', 'TMSOHN', 'NHOSiMe3', 'Me3SiONH'], "[NH]O[Si](C)(C)C"),  # (trimethylsilyloxy)amino
    _extra(['NHOTr', 'TrONH', 'NHOTrt', 'TrtONH'], "[NH]OC(c1ccccc1)(c1ccccc1)c1ccccc1"),  # (trityloxy)amino
    _extra(['NHp-An', 'p-AnHN', 'p-AnNH', 'Np-An', 'p-AnN'], "[NH]c1ccc(OC)cc1"),  # N-(4-methoxyphenyl)amino
    _extra(['NHp-Ns', 'p-NsHN', 'p-NsNH', 'Np-Ns', 'p-NsN'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (p-nosyl))amino
    _extra(['NHp-NZ', 'p-NZHN', 'p-NZNH', 'Np-NZ', 'p-NZN'], "[NH]C(=O)OCc1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzyloxycarbonyl)amino
    _extra(['NHp-TolSO', 'p-TolSOHN', 'p-TolSONH', 'Np-TolSO', 'p-TolSON'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl (Andersen/Davis auxiliary))amino
    _extra(['NHpAn', 'pAnHN', 'pAnNH', 'NpAn', 'pAnN'], "[NH]c1ccc(OC)cc1"),  # N-(4-methoxyphenyl)amino
    _extra(['NHPbf', 'PbfHN', 'PbfNH', 'NPbf', 'PbfN'], "[NH]S(=O)(=O)c1c(C)c(C)c2OC(C)(C)Cc2c1C"),  # N-(2,2,4,6,7-pentamethyl-2,3-dihydrobenzofuran-5-sulfonyl)amino
    _extra(['NHPhF', 'PhFHN', 'PhFNH', 'NPhF', 'PhFN'], "[NH]C1(c2ccccc2)c2ccccc2-c2ccccc21"),  # N-(9-phenylfluoren-9-yl)amino
    _extra(['NHPhFl', 'PhFlHN', 'PhFlNH', 'NPhFl', 'PhFlN'], "[NH]C1(c2ccccc2)c2ccccc2-c2ccccc21"),  # N-(9-phenylfluoren-9-yl)amino
    _extra(['NHPI'], "[O]N1C(=O)c2ccccc2C1=O"),  # N-hydroxyphthalimide as O-attached leaving group (NHPI ester)
    _extra(['NHPI'], "ON1C(=O)c2ccccc2C1=O", 0),  # N-hydroxyphthalimide (reagent molecule)
    _extra(['NHPiv', 'PivHN', 'PivNH', 'NPiv', 'PivN'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl)amino
    _extra(['NHPMB', 'PMBHN', 'PMBNH', 'NPMB', 'PMBN'], "[NH]Cc1ccc(OC)cc1"),  # N-(4-methoxybenzyl)amino
    _extra(['NHPMBM', 'PMBMHN', 'PMBMNH', 'NPMBM', 'PMBMN'], "[NH]COCc1ccc(OC)cc1"),  # N-(4-methoxybenzyloxymethyl)amino
    _extra(['NHPmc', 'PmcHN', 'PmcNH', 'NPmc', 'PmcN'], "[NH]S(=O)(=O)c1c(C)c(C)c2OC(C)(C)CCc2c1C"),  # N-(2,2,5,7,8-pentamethylchroman-6-sulfonyl)amino
    _extra(['NHPMP', 'PMPHN', 'PMPNH', 'NPMP', 'PMPN'], "[NH]c1ccc(OC)cc1"),  # N-(4-methoxyphenyl)amino
    _extra(['NHPms', 'PmsHN', 'PmsNH', 'NPms', 'PmsN'], "[NH]S(=O)(=O)Cc1ccccc1"),  # N-(phenylmethanesulfonyl (benzylsulfonyl))amino
    _extra(['NHpNs', 'pNsHN', 'pNsNH', 'NpNs', 'pNsN'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (p-nosyl))amino
    _extra(['NHpNZ', 'pNZHN', 'pNZNH', 'NpNZ', 'pNZN'], "[NH]C(=O)OCc1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzyloxycarbonyl)amino
    _extra(['NHPoc', 'PocHN', 'PocNH', 'NPoc', 'PocN'], "[NH]C(=O)OCC#C"),  # N-(propargyloxycarbonyl)amino
    _extra(['NHPom', 'PomHN', 'PomNH', 'NPom', 'PomN'], "[NH]COC(=O)C(C)(C)C"),  # N-(pivaloyloxymethyl)amino
    _extra(['NHPOM', 'POMHN', 'POMNH', 'NPOM', 'POMN'], "[NH]COC(=O)C(C)(C)C"),  # N-(pivaloyloxymethyl)amino
    _extra(['NHpTolSO', 'pTolSOHN', 'pTolSONH', 'NpTolSO', 'pTolSON'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl (Andersen/Davis auxiliary))amino
    _extra(['NHPv', 'PvHN', 'PvNH', 'NPv', 'PvN'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl)amino
    _extra(['NHS(O)2CF3', 'NS(O)2CF3'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['NHS(O)2Me', 'NS(O)2Me'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino
    _extra(['NHS(O)2Ph', 'NS(O)2Ph'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino
    _extra(['NHS(O)Mes', 'S(O)MesHN', 'S(O)MesNH', 'NS(O)Mes', 'S(O)MesN'], "[NH]S(=O)c1c(C)cc(C)cc1C"),  # N-(mesitylsulfinyl, group written to the right)amino
    _extra(['NHS(O)tBu', 'S(O)tBuHN', 'S(O)tBuNH', 'NS(O)tBu', 'S(O)tBuN'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary), group written to the right)amino
    _extra(['NHS(O)Tol', 'S(O)TolHN', 'S(O)TolNH', 'NS(O)Tol', 'S(O)TolN'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl, group written to the right)amino
    _extra(['NHSEM', 'SEMHN', 'SEMNH', 'NSEM', 'SEMN'], "[NH]COCC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethoxymethyl)amino
    _extra(['NHSES', 'SESHN', 'SESNH', 'NSES', 'SESN'], "[NH]S(=O)(=O)CC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethanesulfonyl)amino
    _extra(['NHSiMe3', 'NSiMe3'], "[NH][Si](C)(C)C"),  # N-(trimethylsilyl)amino
    _extra(['NHSO2C6H4-2-NO2', 'NSO2C6H4-2-NO2'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4-4-Br', 'NSO2C6H4-4-Br'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4-4-Me', 'NSO2C6H4-4-Me'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4-4-NO2', 'NSO2C6H4-4-NO2'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4-4-OMe', 'NSO2C6H4-4-OMe'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4-p-NO2', 'NSO2C6H4-p-NO2'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4Br', 'NSO2C6H4Br'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4Br-4', 'NSO2C6H4Br-4'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4Br-p', 'NSO2C6H4Br-p'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4CH3', 'NSO2C6H4CH3'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4Me', 'NSO2C6H4Me'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4NO2-2', 'NSO2C6H4NO2-2'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4NO2-4', 'NSO2C6H4NO2-4'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4NO2-o', 'NSO2C6H4NO2-o'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4NO2-p', 'NSO2C6H4NO2-p'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4OMe', 'NSO2C6H4OMe'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4OMe-4', 'NSO2C6H4OMe-4'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2C6H4OMe-p', 'NSO2C6H4OMe-p'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino
    _extra(['NHSO2CF3', 'NSO2CF3'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (formula spelling))amino
    _extra(['NHSO2CH2CH2SiMe3', 'NSO2CH2CH2SiMe3'], "[NH]S(=O)(=O)CC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino
    _extra(['NHSO2CH2CH2TMS', 'NSO2CH2CH2TMS'], "[NH]S(=O)(=O)CC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino
    _extra(['NHSO2CH3', 'NSO2CH3'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino
    _extra(['NHSO2Me', 'NSO2Me'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (formula spelling))amino
    _extra(['NHSO2p-Tol', 'NSO2p-Tol'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['NHSO2Ph', 'NSO2Ph'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino
    _extra(['NHSO2pTol', 'NSO2pTol'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['NHSO2Tol', 'NSO2Tol'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['NHSOMes', 'SOMesHN', 'SOMesNH', 'NSOMes', 'SOMesN'], "[NH]S(=O)c1c(C)cc(C)cc1C"),  # N-(mesitylsulfinyl, group written to the right)amino
    _extra(['NHSOp-Tol', 'SOp-TolHN', 'SOp-TolNH', 'NSOp-Tol', 'SOp-TolN'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl, group written to the right)amino
    _extra(['NHSOpTol', 'SOpTolHN', 'SOpTolNH', 'NSOpTol', 'SOpTolN'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl, group written to the right)amino
    _extra(['NHSOt-Bu', 'SOt-BuHN', 'SOt-BuNH', 'NSOt-Bu', 'SOt-BuN'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary), group written to the right)amino
    _extra(['NHSOtBu', 'SOtBuHN', 'SOtBuNH', 'NSOtBu', 'SOtBuN'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary), group written to the right)amino
    _extra(['NHSOTol', 'SOTolHN', 'SOTolNH', 'NSOTol', 'SOTolN'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl, group written to the right)amino
    _extra(['NHSuc', 'SucHN', 'SucNH', 'NSuc', 'SucN'], "[NH]C(=O)CCC(=O)O"),  # N-(succinyl (3-carboxypropanoyl))amino
    _extra(['NHSucc', 'SuccHN', 'SuccNH', 'NSucc', 'SuccN'], "[NH]C(=O)CCC(=O)O"),  # N-(succinyl (3-carboxypropanoyl))amino
    _extra(['NHt-Boc', 't-BocHN', 't-BocNH', 'Nt-Boc', 't-BocN'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl)amino
    _extra(['NHt-BuSO', 't-BuSOHN', 't-BuSONH', 'Nt-BuSO', 't-BuSON'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary))amino
    _extra(['NHTBDMS', 'TBDMSHN', 'TBDMSNH', 'NTBDMS', 'TBDMSN'], "[NH][Si](C)(C)C(C)(C)C"),  # N-(tert-butyldimethylsilyl)amino
    _extra(['NHTBDPS', 'TBDPSHN', 'TBDPSNH', 'NTBDPS', 'TBDPSN'], "[NH][Si](c1ccccc1)(c1ccccc1)C(C)(C)C"),  # N-(tert-butyldiphenylsilyl)amino
    _extra(['NHtBoc', 'tBocHN', 'tBocNH', 'NtBoc', 'tBocN'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl)amino
    _extra(['NHTBS', 'TBSHN', 'TBSNH', 'NTBS', 'TBSN'], "[NH][Si](C)(C)C(C)(C)C"),  # N-(tert-butyldimethylsilyl)amino
    _extra(['NHtBu(O)S', 'tBu(O)SHN', 'tBu(O)SNH', 'NtBu(O)S', 'tBu(O)SN'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary))amino
    _extra(['NHtBuOS', 'tBuOSHN', 'tBuOSNH', 'NtBuOS', 'tBuOSN'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary))amino
    _extra(['NHtBuS(O)', 'tBuS(O)HN', 'tBuS(O)NH', 'NtBuS(O)', 'tBuS(O)N'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary))amino
    _extra(['NHtBuSO', 'tBuSOHN', 'tBuSONH', 'NtBuSO', 'tBuSON'], "[NH]S(=O)C(C)(C)C"),  # N-(tert-butanesulfinyl (Ellman auxiliary))amino
    _extra(['NHTces', 'TcesHN', 'TcesNH', 'NTces', 'TcesN'], "[NH]S(=O)(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxysulfonyl)amino
    _extra(['NHTeoc', 'TeocHN', 'TeocNH', 'NTeoc', 'TeocN'], "[NH]C(=O)OCC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethoxycarbonyl)amino
    _extra(['NHTES', 'TESHN', 'TESNH', 'NTES', 'TESN'], "[NH][Si](CC)(CC)CC"),  # N-(triethylsilyl)amino
    _extra(['NHTfa', 'TfaHN', 'TfaNH', 'NTfa', 'TfaN'], "[NH]C(=O)C(F)(F)F"),  # N-(trifluoroacetyl)amino
    _extra(['NHTHP', 'THPHN', 'THPNH', 'NTHP', 'THPN'], "[NH]C1CCCCO1"),  # N-(tetrahydropyran-2-yl)amino
    _extra(['NHTIPS', 'TIPSHN', 'TIPSNH', 'NTIPS', 'TIPSN'], "[NH][Si](C(C)C)(C(C)C)C(C)C"),  # N-(triisopropylsilyl)amino
    _extra(['NHTmob', 'TmobHN', 'TmobNH', 'NTmob', 'TmobN'], "[NH]Cc1c(OC)cc(OC)cc1OC"),  # N-(2,4,6-trimethoxybenzyl)amino
    _extra(['NHTMS', 'TMSHN', 'TMSNH', 'NTMS', 'TMSN'], "[NH][Si](C)(C)C"),  # N-(trimethylsilyl)amino
    _extra(['NHTol(O)S', 'Tol(O)SHN', 'Tol(O)SNH', 'NTol(O)S', 'Tol(O)SN'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl (Andersen/Davis auxiliary))amino
    _extra(['NHTolS(O)', 'TolS(O)HN', 'TolS(O)NH', 'NTolS(O)', 'TolS(O)N'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl (Andersen/Davis auxiliary))amino
    _extra(['NHTolSO', 'TolSOHN', 'TolSONH', 'NTolSO', 'TolSON'], "[NH]S(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfinyl (Andersen/Davis auxiliary))amino
    _extra(['NHTos', 'TosHN', 'TosNH', 'NTos', 'TosN'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (tosyl))amino
    _extra(['NHTres', 'TresHN', 'TresNH', 'NTres', 'TresN'], "[NH]S(=O)(=O)CC(F)(F)F"),  # N-(2,2,2-trifluoroethanesulfonyl (tresyl))amino
    _extra(['NHTrisyl', 'TrisylHN', 'TrisylNH', 'NTrisyl', 'TrisylN'], "[NH]S(=O)(=O)c1c(C(C)C)cc(C(C)C)cc1C(C)C"),  # N-(2,4,6-triisopropylbenzenesulfonyl (trisyl))amino
    _extra(['NHTroc', 'TrocHN', 'TrocNH', 'NTroc', 'TrocN'], "[NH]C(=O)OCC(Cl)(Cl)Cl"),  # N-(2,2,2-trichloroethoxycarbonyl)amino
    _extra(['NHVoc', 'VocHN', 'VocNH', 'NVoc', 'VocN'], "[NH]C(=O)OC=C"),  # N-(vinyloxycarbonyl)amino
    _extra(['NHXan', 'XanHN', 'XanNH', 'NXan', 'XanN'], "[NH]C1c2ccccc2Oc2ccccc21"),  # N-(9H-xanthen-9-yl)amino
    _extra(['NHXant', 'XantHN', 'XantNH', 'NXant', 'XantN'], "[NH]C1c2ccccc2Oc2ccccc21"),  # N-(9H-xanthen-9-yl)amino
    _extra(['NHZ', 'ZHN', 'ZNH', 'NZ', 'ZN'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl)amino
    _extra(['Niv-Dde', 'iv-DdeN'], "[N]C(CC(C)C)=C1C(=O)CC(C)(C)CC1=O", 2),  # N-1-(4,4-dimethyl-2,6-dioxocyclohexylidene)-3-methylbutyl nitrogen with two bonds (ring/secondary N)
    _extra(['NivDde', 'ivDdeN'], "[N]C(CC(C)C)=C1C(=O)CC(C)(C)CC1=O", 2),  # N-1-(4,4-dimethyl-2,6-dioxocyclohexylidene)-3-methylbutyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMal', 'MalN', 'Maleimido'], "[N]1C(=O)C=CC1=O"),  # maleimido
    _extra(['NMbh', 'MbhN'], "[N]C(c1ccc(OC)cc1)c1ccc(OC)cc1", 2),  # N-4,4'-dimethoxybenzhydryl nitrogen with two bonds (ring/secondary N)
    _extra(['NMe3+', 'Me3N+', 'N+Me3', 'N(Me)3+', 'N(CH3)3+', 'NMe3', 'Me3N'], "[N+](C)(C)C"),  # trimethylammonio
    _extra(['NMEM', 'MEMN'], "[N]COCCOC", 2),  # N-2-methoxyethoxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMeOZ', 'MeOZN'], "[N]C(=O)OCc1ccc(OC)cc1", 2),  # N-4-methoxybenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMes(O)S', 'Mes(O)SN'], "[N]S(=O)c1c(C)cc(C)cc1C", 2),  # N-2,4,6-trimethylbenzenesulfinyl (mesitylsulfinyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NMesS(O)', 'MesS(O)N'], "[N]S(=O)c1c(C)cc(C)cc1C", 2),  # N-2,4,6-trimethylbenzenesulfinyl (mesitylsulfinyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NMesSO', 'MesSON'], "[N]S(=O)c1c(C)cc(C)cc1C", 2),  # N-2,4,6-trimethylbenzenesulfinyl (mesitylsulfinyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NMmt', 'MmtN'], "[N]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1", 2),  # N-4-methoxytrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NMMT', 'MMTN'], "[N]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1", 2),  # N-4-methoxytrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NMMTr', 'MMTrN'], "[N]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1", 2),  # N-4-methoxytrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NMmtr', 'MmtrN'], "[N]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1", 2),  # N-4-methoxytrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NMoc', 'MocN'], "[N]C(=O)OC", 2),  # N-methoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMOM', 'MOMN'], "[N]COC", 2),  # N-methoxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMoz', 'MozN'], "[N]C(=O)OCc1ccc(OC)cc1", 2),  # N-4-methoxybenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMPM', 'MPMN'], "[N]Cc1ccc(OC)cc1", 2),  # N-4-methoxybenzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMs', 'MsN'], "[NH]S(=O)(=O)C"),  # N-(methanesulfonyl (mesyl))amino
    _extra(['NMs', 'MsN'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (mesyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NMsc', 'MscN'], "[N]C(=O)OCCS(C)(=O)=O", 2),  # N-2-(methylsulfonyl)ethoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMtr', 'MtrN'], "[N]S(=O)(=O)c1c(C)cc(OC)c(C)c1C", 2),  # N-4-methoxy-2,3,6-trimethylbenzenesulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NMts', 'MtsN'], "[N]S(=O)(=O)c1c(C)cc(C)cc1C", 2),  # N-mesitylenesulfonyl (2,4,6-trimethylbenzenesulfonyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NMtt', 'MttN'], "[N]C(c1ccc(C)cc1)(c1ccccc1)c1ccccc1", 2),  # N-4-methyltrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NMttr', 'MttrN'], "[N]C(c1ccc(C)cc1)(c1ccccc1)c1ccccc1", 2),  # N-4-methyltrityl nitrogen with two bonds (ring/secondary N)
    _extra(['NNap', 'NapN'], "[N]Cc1ccc2ccccc2c1", 2),  # N-2-naphthylmethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NNAP', 'NAPN'], "[N]Cc1ccc2ccccc2c1", 2),  # N-2-naphthylmethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NNbs', 'NbsN'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (p-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NNf', 'NfN'], "[N]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", 2),  # N-nonafluorobutanesulfonyl (nonaflyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NNHAc', 'AcHNN', 'AcNHN'], "[N]NC(=O)C", 2),  # 2-(Ac)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHAlloc', 'AllocHNN', 'AllocNHN'], "[N]NC(=O)OCC=C", 2),  # 2-(Alloc)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHBn', 'BnHNN', 'BnNHN'], "[N]NCc1ccccc1", 2),  # 2-(Bn)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHBoc', 'BocHNN', 'BocNHN'], "[N]NC(=O)OC(C)(C)C", 2),  # 2-(Boc)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHBz', 'BzHNN', 'BzNHN'], "[N]NC(=O)c1ccccc1", 2),  # 2-(Bz)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHCbz', 'CbzHNN', 'CbzNHN'], "[N]NC(=O)OCc1ccccc1", 2),  # 2-(Cbz)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHFmoc', 'FmocHNN', 'FmocNHN'], "[N]NC(=O)OCC1c2ccccc2-c2ccccc21", 2),  # 2-(Fmoc)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHMs', 'MsHNN', 'MsNHN'], "[N]NS(=O)(=O)C", 2),  # 2-(Ms)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHNs', 'NsHNN', 'NsNHN'], "[N]NS(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # 2-(Ns)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHPiv', 'PivHNN', 'PivNHN'], "[N]NC(=O)C(C)(C)C", 2),  # 2-(Piv)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHPMB', 'PMBHNN', 'PMBNHN'], "[N]NCc1ccc(OC)cc1", 2),  # 2-(PMB)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHTfa', 'TfaHNN', 'TfaNHN'], "[N]NC(=O)C(F)(F)F", 2),  # 2-(Tfa)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHTroc', 'TrocHNN', 'TrocNHN'], "[N]NC(=O)OCC(Cl)(Cl)Cl", 2),  # 2-(Troc)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHTrt', 'TrtHNN', 'TrtNHN'], "[N]NC(c1ccccc1)(c1ccccc1)c1ccccc1", 2),  # 2-(Trt)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHTs', 'TsHNN', 'TsNHN'], "[N]NS(=O)(=O)c1ccc(C)cc1", 2),  # 2-(Ts)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNHZ', 'ZHNN', 'ZNHN'], "[N]NC(=O)OCc1ccccc1", 2),  # 2-(Z)hydrazin-1-yl nitrogen with two bonds
    _extra(['NNoc', 'NocN'], "[N]C(=O)OCC=Cc1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrocinnamyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NNos', 'NosN'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (o-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NNps', 'NpsN'], "[N]Sc1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrophenylsulfenyl nitrogen with two bonds (ring/secondary N)
    _extra(['NNs', 'NsN'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (o-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NNsc', 'NscN'], "[N]C(=O)OCCS(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-2-(4-nitrophenylsulfonyl)ethoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NNvoc', 'NvocN'], "[N]C(=O)OCc1cc(OC)c(OC)cc1[N+](=O)[O-]", 2),  # N-6-nitroveratryloxycarbonyl (4,5-dimethoxy-2-nitrobenzyloxycarbonyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NNVoc', 'NVocN'], "[N]C(=O)OCc1cc(OC)c(OC)cc1[N+](=O)[O-]", 2),  # N-6-nitroveratryloxycarbonyl (4,5-dimethoxy-2-nitrobenzyloxycarbonyl) nitrogen with two bonds (ring/secondary N)
    _extra(['No-Ns', 'o-NsN'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (o-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NOBn', 'BnON'], "[N]OCc1ccccc1", 2),  # N-(benzyloxy) nitrogen with two bonds
    _extra(['NOBn', 'BnON'], "[NH]OCc1ccccc1"),  # (benzyloxy)amino, one bond
    _extra(['Noc'], "[C](=O)OCC=Cc1ccc([N+](=O)[O-])cc1"),  # 4-nitrocinnamyloxycarbonyl
    _extra(['NOMe', 'MeON', 'NOCH3'], "[N]OC", 2),  # N-methoxy nitrogen with two bonds (oxime-ether/ hydroxamate N)
    _extra(['NOMe', 'MeON', 'NOCH3'], "[NH]OC"),  # methoxyamino, one bond
    _extra(['NoNs', 'oNsN'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (o-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NOTBS', 'TBSON'], "[N]O[Si](C)(C)C(C)(C)C", 2),  # N-(TBS-oxy) nitrogen with two bonds
    _extra(['Np-An', 'p-AnN'], "[N]c1ccc(OC)cc1", 2),  # N-4-methoxyphenyl nitrogen with two bonds (ring/secondary N)
    _extra(['Np-Ns', 'p-NsN'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (p-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['Np-NZ', 'p-NZN'], "[N]C(=O)OCc1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['Np-TolSO', 'p-TolSON'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl (Andersen/Davis auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NpAn', 'pAnN'], "[N]c1ccc(OC)cc1", 2),  # N-4-methoxyphenyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPbf', 'PbfN'], "[N]S(=O)(=O)c1c(C)c(C)c2OC(C)(C)Cc2c1C", 2),  # N-2,2,4,6,7-pentamethyl-2,3-dihydrobenzofuran-5-sulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPh', 'PhN'], "[NH]c1ccccc1"),  # N-(phenyl)amino
    _extra(['NPh', 'PhN'], "[N]c1ccccc1", 2),  # N-phenyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPhF', 'PhFN'], "[N]C1(c2ccccc2)c2ccccc2-c2ccccc21", 2),  # N-9-phenylfluoren-9-yl nitrogen with two bonds (ring/secondary N)
    _extra(['NPhFl', 'PhFlN'], "[N]C1(c2ccccc2)c2ccccc2-c2ccccc21", 2),  # N-9-phenylfluoren-9-yl nitrogen with two bonds (ring/secondary N)
    _extra(['NPhth', 'PhthN', 'NPht', 'PhtN', 'Phthalimido'], "[N]1C(=O)c2ccccc2C1=O"),  # phthalimido (N-phthaloyl amine)
    _extra(['NPiv', 'PivN'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPMB', 'PMBN'], "[N]Cc1ccc(OC)cc1", 2),  # N-4-methoxybenzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPMBM', 'PMBMN'], "[N]COCc1ccc(OC)cc1", 2),  # N-4-methoxybenzyloxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPmc', 'PmcN'], "[N]S(=O)(=O)c1c(C)c(C)c2OC(C)(C)CCc2c1C", 2),  # N-2,2,5,7,8-pentamethylchroman-6-sulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPMP', 'PMPN'], "[N]c1ccc(OC)cc1", 2),  # N-4-methoxyphenyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPms', 'PmsN'], "[N]S(=O)(=O)Cc1ccccc1", 2),  # N-phenylmethanesulfonyl (benzylsulfonyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NpNs', 'pNsN'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (p-nosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NpNZ', 'pNZN'], "[N]C(=O)OCc1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPoc', 'PocN'], "[N]C(=O)OCC#C", 2),  # N-propargyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPom', 'PomN'], "[N]COC(=O)C(C)(C)C", 2),  # N-pivaloyloxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NPOM', 'POMN'], "[N]COC(=O)C(C)(C)C", 2),  # N-pivaloyloxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['Nps'], "[S]c1ccccc1[N+](=O)[O-]"),  # 2-nitrophenylsulfenyl
    _extra(['NpTolSO', 'pTolSON'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl (Andersen/Davis auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NPv', 'PvN'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl nitrogen with two bonds (ring/secondary N)
    _extra(['NS(O)2CF3'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NS(O)2Me'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NS(O)2Ph'], "[N]S(=O)(=O)c1ccccc1", 2),  # N-benzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NS(O)Mes', 'S(O)MesN'], "[N]S(=O)c1c(C)cc(C)cc1C", 2),  # N-mesitylsulfinyl, group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NS(O)tBu', 'S(O)tBuN'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary), group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NS(O)Tol', 'S(O)TolN'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl, group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['Nsc'], "[C](=O)OCCS(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # 2-(4-nitrophenylsulfonyl)ethoxycarbonyl
    _extra(['NSEM', 'SEMN'], "[N]COCC[Si](C)(C)C", 2),  # N-2-(trimethylsilyl)ethoxymethyl nitrogen with two bonds (ring/secondary N)
    _extra(['NSES', 'SESN'], "[N]S(=O)(=O)CC[Si](C)(C)C", 2),  # N-2-(trimethylsilyl)ethanesulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NSiMe3'], "[N][Si](C)(C)C", 2),  # N-trimethylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['NSO2C6H4-2-NO2'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4-4-Br'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4-4-Me'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4-4-NO2'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4-4-OMe'], "[N]S(=O)(=O)c1ccc(OC)cc1", 2),  # N-4-methoxybenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4-p-NO2'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4Br'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4Br-4'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4Br-p'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4CH3'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4Me'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4NO2-2'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4NO2-4'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4NO2-o'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4NO2-p'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4OMe'], "[N]S(=O)(=O)c1ccc(OC)cc1", 2),  # N-4-methoxybenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4OMe-4'], "[N]S(=O)(=O)c1ccc(OC)cc1", 2),  # N-4-methoxybenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2C6H4OMe-p'], "[N]S(=O)(=O)c1ccc(OC)cc1", 2),  # N-4-methoxybenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2CF3'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2CH2CH2SiMe3'], "[N]S(=O)(=O)CC[Si](C)(C)C", 2),  # N-2-(trimethylsilyl)ethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2CH2CH2TMS'], "[N]S(=O)(=O)CC[Si](C)(C)C", 2),  # N-2-(trimethylsilyl)ethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2CH3'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2Me'], "[N]S(=O)(=O)C", 2),  # N-methanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2p-Tol'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2Ph'], "[N]S(=O)(=O)c1ccccc1", 2),  # N-benzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2pTol'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSO2Tol'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['NSOMes', 'SOMesN'], "[N]S(=O)c1c(C)cc(C)cc1C", 2),  # N-mesitylsulfinyl, group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NSOp-Tol', 'SOp-TolN'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl, group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NSOpTol', 'SOpTolN'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl, group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NSOt-Bu', 'SOt-BuN'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary), group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NSOtBu', 'SOtBuN'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary), group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NSOTol', 'SOTolN'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl, group written to the right nitrogen with two bonds (ring/secondary N)
    _extra(['NSu', 'SuN', 'Succinimido'], "[N]1C(=O)CCC1=O"),  # succinimido (N-succinimidyl)
    _extra(['NSuc', 'SucN'], "[N]C(=O)CCC(=O)O", 2),  # N-succinyl (3-carboxypropanoyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NSucc', 'SuccN'], "[N]C(=O)CCC(=O)O", 2),  # N-succinyl (3-carboxypropanoyl) nitrogen with two bonds (ring/secondary N)
    _extra(['Nt-Boc', 't-BocN'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['Nt-BuSO', 't-BuSON'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NTBDMS', 'TBDMSN'], "[N][Si](C)(C)C(C)(C)C", 2),  # N-tert-butyldimethylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTBDPS', 'TBDPSN'], "[N][Si](c1ccccc1)(c1ccccc1)C(C)(C)C", 2),  # N-tert-butyldiphenylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['NtBoc', 'tBocN'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTBS', 'TBSN'], "[N][Si](C)(C)C(C)(C)C", 2),  # N-tert-butyldimethylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['NtBu(O)S', 'tBu(O)SN'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NtBuOS', 'tBuOSN'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NtBuS(O)', 'tBuS(O)N'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NtBuSO', 'tBuSON'], "[N]S(=O)C(C)(C)C", 2),  # N-tert-butanesulfinyl (Ellman auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NTces', 'TcesN'], "[N]S(=O)(=O)OCC(Cl)(Cl)Cl", 2),  # N-2,2,2-trichloroethoxysulfonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTeoc', 'TeocN'], "[N]C(=O)OCC[Si](C)(C)C", 2),  # N-2-(trimethylsilyl)ethoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTES', 'TESN'], "[N][Si](CC)(CC)CC", 2),  # N-triethylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTf', 'TfN'], "[NH]S(=O)(=O)C(F)(F)F"),  # N-(trifluoromethanesulfonyl (triflyl))amino
    _extra(['NTf', 'TfN'], "[N]S(=O)(=O)C(F)(F)F", 2),  # N-trifluoromethanesulfonyl (triflyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NTfa', 'TfaN'], "[N]C(=O)C(F)(F)F", 2),  # N-trifluoroacetyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTHP', 'THPN'], "[N]C1CCCCO1", 2),  # N-tetrahydropyran-2-yl nitrogen with two bonds (ring/secondary N)
    _extra(['NTIPS', 'TIPSN'], "[N][Si](C(C)C)(C(C)C)C(C)C", 2),  # N-triisopropylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTmob', 'TmobN'], "[N]Cc1c(OC)cc(OC)cc1OC", 2),  # N-2,4,6-trimethoxybenzyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTMS', 'TMSN'], "[N][Si](C)(C)C", 2),  # N-trimethylsilyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTol(O)S', 'Tol(O)SN'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl (Andersen/Davis auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NTolS(O)', 'TolS(O)N'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl (Andersen/Davis auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NTolSO', 'TolSON'], "[N]S(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfinyl (Andersen/Davis auxiliary) nitrogen with two bonds (ring/secondary N)
    _extra(['NTos', 'TosN'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (tosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NTr', 'TrN'], "[N]C(c1ccccc1)(c1ccccc1)c1ccccc1", 2),  # N-triphenylmethyl (trityl) nitrogen with two bonds (ring/secondary N)
    _extra(['NTres', 'TresN'], "[N]S(=O)(=O)CC(F)(F)F", 2),  # N-2,2,2-trifluoroethanesulfonyl (tresyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NTrisyl', 'TrisylN'], "[N]S(=O)(=O)c1c(C(C)C)cc(C(C)C)cc1C(C)C", 2),  # N-2,4,6-triisopropylbenzenesulfonyl (trisyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NTroc', 'TrocN'], "[N]C(=O)OCC(Cl)(Cl)Cl", 2),  # N-2,2,2-trichloroethoxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['NTrt', 'TrtN'], "[N]C(c1ccccc1)(c1ccccc1)c1ccccc1", 2),  # N-triphenylmethyl (trityl) nitrogen with two bonds (ring/secondary N)
    _extra(['NTs', 'TsN'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (tosyl) nitrogen with two bonds (ring/secondary N)
    _extra(['NVoc', 'VocN'], "[N]C(=O)OC=C", 2),  # N-vinyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['Nvoc'], "[C](=O)OCc1cc(OC)c(OC)cc1[N+](=O)[O-]"),  # 6-nitroveratryloxycarbonyl (4,5-dimethoxy-2-nitrobenzyloxycarbonyl)
    _extra(['NXan', 'XanN'], "[N]C1c2ccccc2Oc2ccccc21", 2),  # N-9H-xanthen-9-yl nitrogen with two bonds (ring/secondary N)
    _extra(['NXant', 'XantN'], "[N]C1c2ccccc2Oc2ccccc21", 2),  # N-9H-xanthen-9-yl nitrogen with two bonds (ring/secondary N)
    _extra(['NZ', 'ZN'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl nitrogen with two bonds (ring/secondary N)
    _extra(['o-O2NC6H4SO2HN', 'o-O2NC6H4SO2NH'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['o-O2NC6H4SO2N', 'o-O2NC6H4SO2-N'], "[NH]S(=O)(=O)c1ccccc1[N+](=O)[O-]"),  # N-(2-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['o-O2NC6H4SO2N'], "[N]S(=O)(=O)c1ccccc1[N+](=O)[O-]", 2),  # N-2-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['OHCN', 'OHC-N'], "[NH]C=O"),  # N-(formyl (formula spelling))amino, group written left
    _extra(['OHCN'], "[N]C=O", 2),  # N-formyl (formula spelling) nitrogen with two bonds
    _extra(['OHCNBn', 'OHCN(Bn)'], "[N](Cc1ccccc1)C=O"),  # N-benzyl-N-(formyl (formula spelling))amino
    _extra(['OHCNH'], "[NH]C=O"),  # N-(formyl (formula spelling))amino, group written left
    _extra(['OHCNMe', 'OHCN(Me)'], "[N](C)C=O"),  # N-methyl-N-(formyl (formula spelling))amino
    _extra(['ONPhth', 'PhthNO', 'ONPht', 'PhtNO'], "[O]N1C(=O)c2ccccc2C1=O"),  # phthalimido-N-oxy (NHPI ester oxygen)
    _extra(['p-BrC6H4SO2HN', 'p-BrC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-BrC6H4SO2N', 'p-BrC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc(Br)cc1"),  # N-(4-bromobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-BrC6H4SO2N'], "[N]S(=O)(=O)c1ccc(Br)cc1", 2),  # N-4-bromobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['p-MeOC6H4SO2HN', 'p-MeOC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-MeOC6H4SO2N', 'p-MeOC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc(OC)cc1"),  # N-(4-methoxybenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-MeOC6H4SO2N'], "[N]S(=O)(=O)c1ccc(OC)cc1", 2),  # N-4-methoxybenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['p-NO2C6H4SO2HN', 'p-NO2C6H4SO2NH'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-NO2C6H4SO2N', 'p-NO2C6H4SO2-N'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-NO2C6H4SO2N'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['p-O2NC6H4SO2HN', 'p-O2NC6H4SO2NH'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-O2NC6H4SO2N', 'p-O2NC6H4SO2-N'], "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1"),  # N-(4-nitrobenzenesulfonyl (formula spelling))amino, group written left
    _extra(['p-O2NC6H4SO2N'], "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 2),  # N-4-nitrobenzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['p-TolO2SHN', 'p-TolO2SNH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['p-TolO2SN', 'p-TolO2S-N'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['p-TolO2SN'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['p-TolSO2HN', 'p-TolSO2NH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['p-TolSO2N', 'p-TolSO2-N'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['p-TolSO2N'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['p-TolSO2NBn', 'p-TolSO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-benzyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['p-TolSO2NMe', 'p-TolSO2N(Me)'], "[N](C)S(=O)(=O)c1ccc(C)cc1"),  # N-methyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['Pbf'], "[S](=O)(=O)c1c(C)c(C)c2OC(C)(C)Cc2c1C"),  # 2,2,4,6,7-pentamethyl-2,3-dihydrobenzofuran-5-sulfonyl
    _extra(['Ph2CHHN', 'Ph2CHNH', 'Ph2CHN'], "[NH]C(c1ccccc1)c1ccccc1"),  # N-(diphenylmethyl (benzhydryl))amino
    _extra(['Ph2CHN'], "[N]C(c1ccccc1)c1ccccc1", 2),  # N-diphenylmethyl (benzhydryl) nitrogen with two bonds (ring/secondary N)
    _extra(['Ph2HCHN', 'Ph2HCN'], "[NH]C(c1ccccc1)c1ccccc1"),  # N-(diphenylmethyl (benzhydryl))amino
    _extra(['Ph2HCN'], "[N]C(c1ccccc1)c1ccccc1", 2),  # N-diphenylmethyl (benzhydryl) nitrogen with two bonds (ring/secondary N)
    _extra(['Ph3CHN', 'Ph3CNH', 'Ph3CN'], "[NH]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # N-(triphenylmethyl (trityl))amino
    _extra(['Ph3CN'], "[N]C(c1ccccc1)(c1ccccc1)c1ccccc1", 2),  # N-triphenylmethyl (trityl) nitrogen with two bonds (ring/secondary N)
    _extra(['PhC(O)HN', 'PhC(O)NH'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino, group written left
    _extra(['PhC(O)N', 'PhC(O)-N'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino, group written left
    _extra(['PhC(O)N'], "[N]C(=O)c1ccccc1", 2),  # N-benzoyl (formula spelling) nitrogen with two bonds
    _extra(['PhCH2HN', 'PhCH2NH', 'PhCH2N'], "[NH]Cc1ccccc1"),  # N-(benzyl)amino
    _extra(['PhCH2N'], "[N]Cc1ccccc1", 2),  # N-benzyl nitrogen with two bonds (ring/secondary N)
    _extra(['PhCH2O2CHN', 'PhCH2O2CNH'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['PhCH2O2CN', 'PhCH2O2C-N'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['PhCH2O2CN'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['PhCH2OC(O)HN', 'PhCH2OC(O)NH'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['PhCH2OC(O)N', 'PhCH2OC(O)-N'], "[NH]C(=O)OCc1ccccc1"),  # N-(benzyloxycarbonyl (formula spelling))amino, group written left
    _extra(['PhCH2OC(O)N'], "[N]C(=O)OCc1ccccc1", 2),  # N-benzyloxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['PhCMe2HN', 'PhCMe2NH', 'PhCMe2N'], "[NH]C(C)(C)c1ccccc1"),  # N-(cumyl (2-phenylpropan-2-yl))amino
    _extra(['PhCMe2N'], "[N]C(C)(C)c1ccccc1", 2),  # N-cumyl (2-phenylpropan-2-yl) nitrogen with two bonds (ring/secondary N)
    _extra(['PhCOHN', 'PhCONH'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino, group written left
    _extra(['PhCON', 'PhCO-N'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino, group written left
    _extra(['PhCON'], "[N]C(=O)c1ccccc1", 2),  # N-benzoyl (formula spelling) nitrogen with two bonds
    _extra(['PhCONBn', 'PhCON(Bn)'], "[N](Cc1ccccc1)C(=O)c1ccccc1"),  # N-benzyl-N-(benzoyl (formula spelling))amino
    _extra(['PhCONMe', 'PhCON(Me)'], "[N](C)C(=O)c1ccccc1"),  # N-methyl-N-(benzoyl (formula spelling))amino
    _extra(['PhF', 'PhFl', '9-PhF'], "[C]1(c2ccccc2)c2ccccc2-c2ccccc21"),  # 9-phenylfluoren-9-yl
    _extra(['PhH2CHN', 'PhH2CN'], "[NH]Cc1ccccc1"),  # N-(benzyl)amino
    _extra(['PhH2CN'], "[N]Cc1ccccc1", 2),  # N-benzyl nitrogen with two bonds (ring/secondary N)
    _extra(['PhMe2CHN', 'PhMe2CNH', 'PhMe2CN'], "[NH]C(C)(C)c1ccccc1"),  # N-(cumyl (2-phenylpropan-2-yl))amino
    _extra(['PhMe2CN'], "[N]C(C)(C)c1ccccc1", 2),  # N-cumyl (2-phenylpropan-2-yl) nitrogen with two bonds (ring/secondary N)
    _extra(['PhO2SHN', 'PhO2SNH'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino, group written left
    _extra(['PhO2SN', 'PhO2S-N'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino, group written left
    _extra(['PhO2SN'], "[N]S(=O)(=O)c1ccccc1", 2),  # N-benzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['PhOCHN', 'PhOCNH'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino, group written left
    _extra(['PhOCN', 'PhOC-N'], "[NH]C(=O)c1ccccc1"),  # N-(benzoyl (formula spelling))amino, group written left
    _extra(['PhOCN'], "[N]C(=O)c1ccccc1", 2),  # N-benzoyl (formula spelling) nitrogen with two bonds
    _extra(['PhS(O)2HN', 'PhS(O)2NH'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino, group written left
    _extra(['PhS(O)2N', 'PhS(O)2-N'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino, group written left
    _extra(['PhS(O)2N'], "[N]S(=O)(=O)c1ccccc1", 2),  # N-benzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['PhSO2HN', 'PhSO2NH'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino, group written left
    _extra(['PhSO2N', 'PhSO2-N'], "[NH]S(=O)(=O)c1ccccc1"),  # N-(benzenesulfonyl (formula spelling))amino, group written left
    _extra(['PhSO2N'], "[N]S(=O)(=O)c1ccccc1", 2),  # N-benzenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['PhSO2NBn', 'PhSO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)c1ccccc1"),  # N-benzyl-N-(benzenesulfonyl (formula spelling))amino
    _extra(['PhSO2NMe', 'PhSO2N(Me)'], "[N](C)S(=O)(=O)c1ccccc1"),  # N-methyl-N-(benzenesulfonyl (formula spelling))amino
    _extra(['Phth', 'Pht', 'Phthaloyl'], "[C](=O)c1ccccc1[C]=O", 2),  # phthaloyl (diacyl, drawn in-line on an explicit N)
    _extra(['PMBM'], "[CH2]OCc1ccc(OC)cc1"),  # 4-methoxybenzyloxymethyl
    _extra(['Pmc'], "[S](=O)(=O)c1c(C)c(C)c2OC(C)(C)CCc2c1C"),  # 2,2,5,7,8-pentamethylchroman-6-sulfonyl
    _extra(['Pms'], "[S](=O)(=O)Cc1ccccc1"),  # phenylmethanesulfonyl (benzylsulfonyl)
    _extra(['pNZ', 'p-NZ'], "[C](=O)OCc1ccc([N+](=O)[O-])cc1"),  # 4-nitrobenzyloxycarbonyl
    _extra(['Poc'], "[C](=O)OCC#C"),  # propargyloxycarbonyl
    _extra(['Pom', 'POM'], "[CH2]OC(=O)C(C)(C)C"),  # pivaloyloxymethyl
    _extra(['pTolO2SHN', 'pTolO2SNH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['pTolO2SN', 'pTolO2S-N'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['pTolO2SN'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['pTolSO2HN', 'pTolSO2NH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['pTolSO2N', 'pTolSO2-N'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['pTolSO2N'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['pTolSO2NBn', 'pTolSO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-benzyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['pTolSO2NMe', 'pTolSO2N(Me)'], "[N](C)S(=O)(=O)c1ccc(C)cc1"),  # N-methyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['S(O)Mes', 'SOMes'], "[S](=O)c1c(C)cc(C)cc1C"),  # mesitylsulfinyl, group written to the right
    _extra(['S(O)tBu', 'S(O)t-Bu', 'SOtBu', 'SOt-Bu'], "[S](=O)C(C)(C)C"),  # tert-butanesulfinyl (Ellman auxiliary), group written to the right
    _extra(['S(O)Tol', 'S(O)p-Tol', 'S(O)pTol', 'SOTol', 'SOp-Tol', 'SOpTol'], "[S](=O)c1ccc(C)cc1"),  # p-toluenesulfinyl, group written to the right
    _extra(['SES'], "[S](=O)(=O)CC[Si](C)(C)C"),  # 2-(trimethylsilyl)ethanesulfonyl
    _extra(['Suc', 'Succ', 'Succinyl'], "[C](=O)CC[C]=O", 2),  # succinyl diacyl (succinimide drawn on an explicit N)
    _extra(['Succ'], "[C](=O)CCC(=O)O"),  # succinyl (3-carboxypropanoyl)
    _extra(['t-Boc', 'tBoc'], "[C](=O)OC(C)(C)C"),  # tert-butoxycarbonyl
    _extra(['t-BuCOHN', 't-BuCONH'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['t-BuCON', 't-BuCO-N'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['t-BuCON'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl (formula spelling) nitrogen with two bonds
    _extra(['t-BuO2CHN', 't-BuO2CNH'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['t-BuO2CN', 't-BuO2C-N'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['t-BuO2CN'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['tBuC(O)HN', 'tBuC(O)NH'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['tBuC(O)N', 'tBuC(O)-N'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['tBuC(O)N'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl (formula spelling) nitrogen with two bonds
    _extra(['tBuCOHN', 'tBuCONH'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['tBuCON', 'tBuCO-N'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['tBuCON'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl (formula spelling) nitrogen with two bonds
    _extra(['tBuCONBn', 'tBuCON(Bn)'], "[N](Cc1ccccc1)C(=O)C(C)(C)C"),  # N-benzyl-N-(pivaloyl (formula spelling))amino
    _extra(['tBuCONMe', 'tBuCON(Me)'], "[N](C)C(=O)C(C)(C)C"),  # N-methyl-N-(pivaloyl (formula spelling))amino
    _extra(['tBuO2CHN', 'tBuO2CNH'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['tBuO2CN', 'tBuO2C-N'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['tBuO2CN'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['tBuO2CNBn', 'tBuO2CN(Bn)'], "[N](Cc1ccccc1)C(=O)OC(C)(C)C"),  # N-benzyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['tBuO2CNMe', 'tBuO2CN(Me)'], "[N](C)C(=O)OC(C)(C)C"),  # N-methyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['tBuOC(O)HN', 'tBuOC(O)NH'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['tBuOC(O)N', 'tBuOC(O)-N'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['tBuOC(O)N'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['tBuOCHN', 'tBuOCNH'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['tBuOCN', 'tBuOC-N'], "[NH]C(=O)C(C)(C)C"),  # N-(pivaloyl (formula spelling))amino, group written left
    _extra(['tBuOCN'], "[N]C(=O)C(C)(C)C", 2),  # N-pivaloyl (formula spelling) nitrogen with two bonds
    _extra(['tBuOOCHN', 'tBuOOCNH'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['tBuOOCN', 'tBuOOC-N'], "[NH]C(=O)OC(C)(C)C"),  # N-(tert-butoxycarbonyl (formula spelling))amino, group written left
    _extra(['tBuOOCN'], "[N]C(=O)OC(C)(C)C", 2),  # N-tert-butoxycarbonyl (formula spelling) nitrogen with two bonds
    _extra(['tBuOOCNBn', 'tBuOOCN(Bn)'], "[N](Cc1ccccc1)C(=O)OC(C)(C)C"),  # N-benzyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['tBuOOCNMe', 'tBuOOCN(Me)'], "[N](C)C(=O)OC(C)(C)C"),  # N-methyl-N-(tert-butoxycarbonyl (formula spelling))amino
    _extra(['tBuS(O)', 't-BuS(O)', 'tBuSO', 't-BuSO', 'tBu(O)S', 'tBuOS'], "[S](=O)C(C)(C)C"),  # tert-butanesulfinyl (Ellman auxiliary)
    _extra(['Tces'], "[S](=O)(=O)OCC(Cl)(Cl)Cl"),  # 2,2,2-trichloroethoxysulfonyl
    _extra(['Tfa'], "[C](=O)C(F)(F)F"),  # trifluoroacetyl
    _extra(['Tmob', '2,4,6-TMB'], "[CH2]c1c(OC)cc(OC)cc1OC"),  # 2,4,6-trimethoxybenzyl
    _extra(['TMSCH2CH2SO2HN', 'TMSCH2CH2SO2NH'], "[NH]S(=O)(=O)CC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino, group written left
    _extra(['TMSCH2CH2SO2N', 'TMSCH2CH2SO2-N'], "[NH]S(=O)(=O)CC[Si](C)(C)C"),  # N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino, group written left
    _extra(['TMSCH2CH2SO2N'], "[N]S(=O)(=O)CC[Si](C)(C)C", 2),  # N-2-(trimethylsilyl)ethanesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['TMSCH2CH2SO2NBn', 'TMSCH2CH2SO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)CC[Si](C)(C)C"),  # N-benzyl-N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino
    _extra(['TMSCH2CH2SO2NMe', 'TMSCH2CH2SO2N(Me)'], "[N](C)S(=O)(=O)CC[Si](C)(C)C"),  # N-methyl-N-(2-(trimethylsilyl)ethanesulfonyl (formula spelling))amino
    _extra(['TolO2SHN', 'TolO2SNH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['TolO2SN', 'TolO2S-N'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['TolO2SN'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['TolS(O)', 'p-TolS(O)', 'pTolS(O)', 'TolSO', 'p-TolSO', 'pTolSO', 'Tol(O)S', 'p-Tol(O)S'], "[S](=O)c1ccc(C)cc1"),  # p-toluenesulfinyl (Andersen/Davis auxiliary)
    _extra(['TolSO2HN', 'TolSO2NH'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['TolSO2N', 'TolSO2-N'], "[NH]S(=O)(=O)c1ccc(C)cc1"),  # N-(p-toluenesulfonyl (formula spelling))amino, group written left
    _extra(['TolSO2N'], "[N]S(=O)(=O)c1ccc(C)cc1", 2),  # N-p-toluenesulfonyl (formula spelling) nitrogen with two bonds
    _extra(['TolSO2NBn', 'TolSO2N(Bn)'], "[N](Cc1ccccc1)S(=O)(=O)c1ccc(C)cc1"),  # N-benzyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['TolSO2NMe', 'TolSO2N(Me)'], "[N](C)S(=O)(=O)c1ccc(C)cc1"),  # N-methyl-N-(p-toluenesulfonyl (formula spelling))amino
    _extra(['Tos'], "[S](=O)(=O)c1ccc(C)cc1"),  # p-toluenesulfonyl (tosyl)
    _extra(['Tres'], "[S](=O)(=O)CC(F)(F)F"),  # 2,2,2-trifluoroethanesulfonyl (tresyl)
    _extra(['Trisyl'], "[S](=O)(=O)c1c(C(C)C)cc(C(C)C)cc1C(C)C"),  # 2,4,6-triisopropylbenzenesulfonyl (trisyl)
    _extra(['TrNH', 'NTr', 'TrN'], "[NH]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # N-(triphenylmethyl (trityl))amino
    _extra(['TrtNH', 'NTrt', 'TrtN'], "[NH]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # N-(triphenylmethyl (trityl))amino
    _extra(['Voc'], "[C](=O)OC=C"),  # vinyloxycarbonyl
    _extra(['Xan', 'Xant'], "[CH]1c2ccccc2Oc2ccccc21"),  # 9H-xanthen-9-yl
    # alkyl
    _extra(['(CH2)10'], "[CH2]CCCCCCCC[CH2]", 2),  # decamethylene (polymethylene chain, 10 CH2)
    _extra(['(CH2)11'], "[CH2]CCCCCCCCC[CH2]", 2),  # undecamethylene (polymethylene chain, 11 CH2)
    _extra(['(CH2)12'], "[CH2]CCCCCCCCCC[CH2]", 2),  # dodecamethylene (polymethylene chain, 12 CH2)
    _extra(['(CH2)3', 'CH2CH2CH2'], "[CH2]C[CH2]", 2),  # propane-1,3-diyl
    _extra(['(CH2)3Ph', 'Ph(CH2)3', 'CH2CH2CH2Ph', 'PhCH2CH2CH2'], "[CH2]CCc1ccccc1"),  # 3-phenylpropyl
    _extra(['(CH2)4', 'CH2CH2CH2CH2'], "[CH2]CC[CH2]", 2),  # tetramethylene (polymethylene chain, 4 CH2)
    _extra(['(CH2)4Ph', 'Ph(CH2)4'], "[CH2]CCCc1ccccc1"),  # 4-phenylbutyl
    _extra(['(CH2)5'], "[CH2]CCC[CH2]", 2),  # pentamethylene (polymethylene chain, 5 CH2)
    _extra(['(CH2)6'], "[CH2]CCCC[CH2]", 2),  # hexamethylene (polymethylene chain, 6 CH2)
    _extra(['(CH2)7'], "[CH2]CCCCC[CH2]", 2),  # heptamethylene (polymethylene chain, 7 CH2)
    _extra(['(CH2)8'], "[CH2]CCCCCC[CH2]", 2),  # octamethylene (polymethylene chain, 8 CH2)
    _extra(['(CH2)9'], "[CH2]CCCCCCC[CH2]", 2),  # nonamethylene (polymethylene chain, 9 CH2)
    _extra(['1,3-butadienyl', 'butadienyl', 'Butadienyl'], "[CH]=CC=C"),  # buta-1,3-dienyl
    _extra(['1-Adm', 'Adm'], "[C]12CC3CC(CC(C3)C1)C2"),  # 1-adamantyl
    _extra(['1-butenyl', '1-Butenyl', 'CHCHEt', 'CHCHCH2CH3', 'EtCHCH', 'CH=CHEt'], "[CH]=CCC"),  # 1-butenyl
    _extra(['1-butynyl', 'CCEt', 'EtCC', 'CCCH2CH3'], "[C]#CCC"),  # 1-butynyl
    _extra(['1-cyclohexenyl', '1-Cyclohexenyl', 'cyclohexenyl', 'Cyclohexenyl'], "[C]1=CCCCC1"),  # cyclohex-1-en-1-yl
    _extra(['1-cyclopentenyl', '1-Cyclopentenyl', 'cyclopentenyl', 'Cyclopentenyl'], "[C]1=CCCC1"),  # cyclopent-1-en-1-yl
    _extra(['1-MecPr', '1-Me-cPr'], "[C]1(C)CC1"),  # 1-methylcyclopropyl
    _extra(['1-methylallyl', 'α-methylallyl', 'CH(CH3)CHCH2', 'CH(Me)CHCH2', 'CHMeVin', 'CH(Me)Vi'], "[CH](C)C=C"),  # 1-methylallyl (but-3-en-2-yl)
    _extra(['1-propenyl', '1-Propenyl', 'propenyl', 'Propenyl', 'CHCHCH3', 'CHCHMe', 'MeCHCH', 'H3CCHCH', 'CH=CHCH3', 'CH=CHMe'], "[CH]=CC"),  # 1-propenyl
    _extra(['1-propynyl', 'propynyl', 'Propynyl', 'CCMe', 'MeCC', 'CCCH3', 'H3CCC'], "[C]#CC"),  # 1-propynyl
    _extra(['2-Ad', '2-Adm'], "[CH]1C2CC3CC(C2)CC1C3"),  # 2-adamantyl
    _extra(['2-butynyl', 'CH2CCMe', 'CH2CCCH3', 'MeCCCH2'], "[CH2]C#CC"),  # 2-butynyl
    _extra(['2-cyclohexenyl', '2-Cyclohexenyl'], "[CH]1C=CCCC1"),  # cyclohex-2-en-1-yl
    _extra(['2-cyclopentenyl', '2-Cyclopentenyl'], "[CH]1C=CCC1"),  # cyclopent-2-en-1-yl
    _extra(['2-EtHex', '2-ethylhexyl', '2-Ethylhexyl', 'CH2CH(Et)Bu', 'CH2CH(Et)nBu', 'CH2CH(C2H5)C4H9', 'CH2CH(Et)C4H9', 'CH2CH(CH2CH3)CH2CH2CH2CH3', 'CH2CH(CH2CH3)(CH2)3CH3', 'BuCH(Et)CH2', 'nBuCH(Et)CH2', '2-EH'], "[CH2]C(CC)CCCC"),  # 2-ethylhexyl
    _extra(['2-Hex', 'CH(CH3)C4H9', 'CH(Me)nBu', 'CH(Me)Bu', 'CHMeBu'], "[CH](C)CCCC"),  # 2-hexyl (1-methylpentyl)
    _extra(['2-norbornyl', 'norbornyl'], "[CH]1CC2CCC1C2"),  # 2-norbornyl (bicyclo[2.2.1]hept-2-yl)
    _extra(['2-Oct', 'sec-Oct', 'CH(CH3)C6H13', 'CHMeHex', 'CH(Me)Hex', 'CH(Me)nHex'], "[CH](C)CCCCCC"),  # 2-octyl (1-methylheptyl)
    _extra(['2-Pent', 'sec-Pent', 'sAm', 's-Am', 'sec-Am', 'CH(CH3)C3H7', 'CH(Me)nPr', 'CH(Me)Pr', 'CHMePr', 'CH(CH3)CH2CH2CH3'], "[CH](C)CCC"),  # 2-pentyl (1-methylbutyl, sec-amyl)
    _extra(['3-cyclohexenyl', '3-Cyclohexenyl'], "[CH]1CC=CCC1"),  # cyclohex-3-en-1-yl
    _extra(['3-Hex', 'CH(Et)Pr', 'CH(Et)nPr', 'CH(C2H5)C3H7'], "[CH](CC)CCC"),  # 3-hexyl (1-ethylbutyl)
    _extra(['3-Pent', 'CHEt2', 'Et2CH', 'CH(Et)2', 'CH(CH2CH3)2', '(CH2CH3)2CH', '(C2H5)2CH', 'CH(C2H5)2'], "[CH](CC)CC"),  # 3-pentyl (1-ethylpropyl)
    _extra(['All', 'allyl', 'Allyl', 'CH2CHCH2', 'H2CCHCH2', 'CH2CH=CH2', 'H2C=CHCH2'], "[CH2]C=C"),  # allyl
    _extra(['allenyl', 'Allenyl', 'CHCCH2', 'H2CCCH', 'CH=C=CH2'], "[CH]=C=C"),  # allenyl (propa-1,2-dienyl)
    _extra(['Am'], "[CH2]CCCC"),  # n-pentyl (amyl)
    _extra(['bicyclo[1.1.1]pentyl'], "[C]12CC(C2)C1"),  # bicyclo[1.1.1]pent-1-yl
    _extra(['Bu-s', '2-Bu', 's-C4H9', 'sC4H9', 'sec-C4H9', 'CHMeEt', 'EtMeCH', 'MeEtCH', 'CH(CH3)CH2CH3', 'CH(Me)Et', 'CH(CH3)Et', 'CH(CH3)C2H5'], "[CH](C)CC"),  # sec-butyl
    _extra(['Bui', 'Bu-i', 'isoBu', 'iso-Bu', 'i-C4H9', 'iC4H9', 'iso-C4H9', 'CH2CHMe2', 'Me2CHCH2', 'CH2CH(CH3)2', '(CH3)2CHCH2', '(H3C)2CHCH2', 'CH2iPr', 'iPrCH2', 'iPrH2C'], "[CH2]C(C)C"),  # isobutyl
    _extra(['Bun', 'Bu-n', 'n-C4H9', 'nC4H9', 'H9C4', 'CH2CH2CH2CH3', '(CH2)3CH3'], "[CH2]CCC"),  # n-butyl
    _extra(['C11H23', 'n-C11H23', 'nC11H23', 'H23C11', '(CH2)10CH3'], "[CH2]CCCCCCCCCC"),  # n-undecyl
    _extra(['C12H25', 'n-C12H25', 'nC12H25', 'H25C12', '(CH2)11CH3'], "[CH2]CCCCCCCCCCC"),  # n-dodecyl
    _extra(['C13H27', 'n-C13H27', 'nC13H27', 'H27C13', '(CH2)12CH3'], "[CH2]CCCCCCCCCCCC"),  # n-tridecyl
    _extra(['C14H29', 'n-C14H29', 'nC14H29', 'H29C14', '(CH2)13CH3'], "[CH2]CCCCCCCCCCCCC"),  # n-tetradecyl
    _extra(['C15H31', 'n-C15H31', 'nC15H31', 'H31C15', '(CH2)14CH3'], "[CH2]CCCCCCCCCCCCCC"),  # n-pentadecyl
    _extra(['C16H33', 'n-C16H33', 'nC16H33', 'H33C16', '(CH2)15CH3'], "[CH2]CCCCCCCCCCCCCCC"),  # n-hexadecyl
    _extra(['C17H35', 'n-C17H35', 'nC17H35', 'H35C17', '(CH2)16CH3'], "[CH2]CCCCCCCCCCCCCCCC"),  # n-heptadecyl
    _extra(['C18H37', 'n-C18H37', 'nC18H37', 'H37C18', '(CH2)17CH3'], "[CH2]CCCCCCCCCCCCCCCCC"),  # n-octadecyl
    _extra(['C19H39', 'n-C19H39', 'nC19H39', 'H39C19', '(CH2)18CH3'], "[CH2]CCCCCCCCCCCCCCCCCC"),  # n-nonadecyl
    _extra(['C20H41', 'n-C20H41', 'nC20H41', 'H41C20', '(CH2)19CH3'], "[CH2]CCCCCCCCCCCCCCCCCCC"),  # n-eicosyl
    _extra(['C21H43', 'n-C21H43', 'nC21H43', 'H43C21', '(CH2)20CH3'], "[CH2]CCCCCCCCCCCCCCCCCCCC"),  # n-heneicosyl
    _extra(['C22H45', 'n-C22H45', 'nC22H45', 'H45C22', '(CH2)21CH3'], "[CH2]CCCCCCCCCCCCCCCCCCCCC"),  # n-docosyl
    _extra(['C2D5', 'D5C2', 'CD2CD3', 'D3CD2C'], "[C]([2H])([2H])C([2H])([2H])[2H]"),  # pentadeuteroethyl
    _extra(['C3D7', 'D7C3', 'n-C3D7', 'CD2CD2CD3', 'D3CD2CD2C'], "[C]([2H])([2H])C([2H])([2H])C([2H])([2H])[2H]"),  # heptadeuteropropyl
    _extra(['C6D5', 'D5C6'], "[c]1c([2H])c([2H])c([2H])c([2H])c1[2H]"),  # pentadeuterophenyl
    _extra(['C6D6'], "[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]", 0),  # benzene-d6
    _extra(['CD', 'DC'], "[C][2H]", 3),  # deuteromethine
    _extra(['CD2', 'D2C'], "[C]([2H])[2H]", 2),  # dideuteromethylene
    _extra(['CD2H', 'CHD2', 'D2HC', 'HD2C', 'D2CH'], "[CH]([2H])[2H]"),  # dideuteromethyl
    _extra(['CD2Ph', 'PhCD2'], "[C]([2H])([2H])c1ccccc1"),  # α,α-dideuterobenzyl
    _extra(['CD3', 'D3C'], "[C]([2H])([2H])[2H]"),  # trideuteromethyl
    _extra(['CD3CN', 'MeCN-d3'], "[2H]C([2H])([2H])C#N", 0),  # acetonitrile-d3
    _extra(['CD3O', 'OCD3', 'D3CO'], "[O]C([2H])([2H])[2H]"),  # trideuteromethoxy
    _extra(['CD3OD', 'MeOD', 'MeOH-d4'], "[2H]OC([2H])([2H])[2H]", 0),  # methanol-d4
    _extra(['CD3OH', 'MeOH-d3'], "OC([2H])([2H])[2H]", 0),  # methanol-d3
    _extra(['CDCl3'], "[2H]C(Cl)(Cl)Cl", 0),  # chloroform-d
    _extra(['CDO', 'OCD'], "[C]([2H])=O"),  # deuteroformyl
    _extra(['cDodec', 'c-Dodec', 'cC12H23', 'c-C12H23'], "[CH]1CCCCCCCCCCC1"),  # cyclododecyl
    _extra(['CEt2', 'Et2C', 'C(Et)2', 'C(C2H5)2', '(C2H5)2C', 'C(CH2CH3)2'], "[C](CC)CC", 2),  # pentane-3,3-diyl (gem-diethyl carbon)
    _extra(['CH2Ad', 'AdCH2', 'AdH2C'], "[CH2]C12CC3CC(CC(C3)C1)C2"),  # 1-adamantylmethyl
    _extra(['CH2cBu', 'cBuCH2', 'cBuH2C'], "[CH2]C1CCC1"),  # cyclobutylmethyl
    _extra(['CH2CCTMS', 'TMSCCCH2', 'CH2CCSiMe3', 'Me3SiCCCH2'], "[CH2]C#C[Si](C)(C)C"),  # 3-(trimethylsilyl)prop-2-ynyl
    _extra(['CH2CH(CH3)CH2CH3', 'CH2CH(Me)Et', 'CH2CHMeEt', 'CH2sBu', 'sBuCH2', 'EtMeCHCH2', 'EtCH(Me)CH2'], "[CH2]C(C)CC"),  # 2-methylbutyl
    _extra(['CH2CH2Cy', 'CyCH2CH2', 'Cy(CH2)2', '(CH2)2Cy'], "[CH2]CC1CCCCC1"),  # 2-cyclohexylethyl
    _extra(['CH2CH2Ph', 'PhCH2CH2', 'Ph(CH2)2', '(CH2)2Ph', 'CH2Bn', 'BnCH2', 'phenethyl', 'Phenethyl'], "[CH2]Cc1ccccc1"),  # phenethyl (2-phenylethyl)
    _extra(['CH2CH2tBu', 'tBuCH2CH2', 'tBu(CH2)2', '(CH2)2tBu', 'CH2CH2CMe3', 'Me3CCH2CH2', 'CH2CH2C(CH3)3', '(CH3)3CCH2CH2'], "[CH2]CC(C)(C)C"),  # 3,3-dimethylbutyl (neohexyl)
    _extra(['CH2CH3', 'H5C2'], "[CH2]C"),  # ethyl
    _extra(['CH2CHMe', 'CH2CH(CH3)', 'CH2CH(Me)'], "[CH2][CH]C", 2),  # propane-1,2-diyl (left CH2)
    _extra(['CH2CMe2', 'CH2C(CH3)2', 'CH2C(Me)2'], "[CH2][C](C)C", 2),  # 2-methylpropane-1,2-diyl (left CH2)
    _extra(['CH2cPent', 'cPentCH2', 'CH2Cyp', 'CypCH2'], "[CH2]C1CCCC1"),  # cyclopentylmethyl
    _extra(['CH2cPr', 'cPrCH2', 'cPrH2C', 'CH2cyPr'], "[CH2]C1CC1"),  # cyclopropylmethyl
    _extra(['CH2Cy', 'CyCH2', 'CyH2C', 'CH2cHex', 'cHexCH2', 'CH2C6H11', 'C6H11CH2'], "[CH2]C1CCCCC1"),  # cyclohexylmethyl
    _extra(['CH2D', 'DH2C', 'DCH2'], "[CH2][2H]"),  # monodeuteromethyl
    _extra(['CHD', 'CDH'], "[CH][2H]", 2),  # monodeuteromethylene
    _extra(['cHept', 'c-Hept', 'cyHept', 'cHep', 'c-Hep', 'cyclo-Hept', 'cC7H13', 'c-C7H13', 'cyclo-C7H13'], "[CH]1CCCCCC1"),  # cycloheptyl
    _extra(['CHEt', 'EtCH', 'CH(Et)', 'CH(C2H5)', 'CHC2H5', 'CH(CH2CH3)'], "[CH]CC", 2),  # propane-1,1-diyl (ethyl-substituted methylene)
    _extra(['CHiPr', 'iPrCH', 'CH(iPr)', 'CH(CH(CH3)2)', 'CHCHMe2'], "[CH]C(C)C", 2),  # 2-methylpropane-1,1-diyl (isopropyl-substituted methylene)
    _extra(['CHMe', 'MeCH', 'MeHC', 'CH(CH3)', 'CHCH3', 'H3CCH', 'CH(Me)'], "[CH]C", 2),  # ethane-1,1-diyl (methyl-substituted methylene)
    _extra(['CHMeCH2', 'CH(CH3)CH2', 'CH(Me)CH2', 'MeCHCH2'], "[CH](C)[CH2]", 2),  # propane-1,2-diyl (left CHMe)
    _extra(['CHMePh', 'PhMeCH', 'CH(CH3)Ph', 'CH(Me)Ph', 'PhCH(CH3)', 'PhCHMe', 'CH(Ph)CH3', 'CH(Ph)Me', 'MeCHPh'], "[CH](C)c1ccccc1"),  # 1-phenylethyl (α-methylbenzyl)
    _extra(['CHPh2', 'Ph2CH', 'CH(Ph)2', 'benzhydryl', 'Benzhydryl', 'Bzh'], "[CH](c1ccccc1)c1ccccc1"),  # benzhydryl (diphenylmethyl)
    _extra(['CHtBu', 'tBuCH', 'CH(tBu)', 'CH(C(CH3)3)', 'CHCMe3'], "[CH]C(C)(C)C", 2),  # 2,2-dimethylpropane-1,1-diyl (tert-butyl-substituted methylene)
    _extra(['cinnamyl', 'Cinnamyl', 'CH2CHCHPh', 'PhCHCHCH2', 'CH2CH=CHPh'], "[CH2]C=Cc1ccccc1"),  # cinnamyl (3-phenylprop-2-enyl)
    _extra(['CMe2', 'Me2C', 'C(CH3)2', '(CH3)2C', '(H3C)2C', 'C(Me)2'], "[C](C)C", 2),  # propane-2,2-diyl (gem-dimethyl carbon)
    _extra(['CMe2CH2', 'C(CH3)2CH2', 'Me2CCH2', 'C(Me)2CH2'], "[C](C)(C)[CH2]", 2),  # 2-methylpropane-1,2-diyl (left CMe2)
    _extra(['CMeEt', 'C(Me)Et', 'C(Me)(Et)', 'EtMeC', 'C(CH3)(CH2CH3)', 'C(CH3)Et', 'MeEtC'], "[C](C)CC", 2),  # butane-2,2-diyl
    _extra(['CO2CD3', 'COOCD3', 'D3CO2C', 'D3COOC'], "[C](=O)OC([2H])([2H])[2H]"),  # trideuteromethyl ester
    _extra(['CO2D', 'COOD', 'DO2C', 'DOOC'], "[C](=O)O[2H]"),  # carboxyl-d (deuterated carboxylic acid)
    _extra(['cOct', 'c-Oct', 'cyOct', 'cyclo-Oct', 'cC8H15', 'c-C8H15', 'cyclo-C8H15'], "[CH]1CCCCCCC1"),  # cyclooctyl
    _extra(['Cp', 'C5H5', 'H5C5'], "[CH]1C=CC=C1"),  # cyclopentadienyl
    _extra(['Cp*', 'C5Me5', 'Me5C5'], "[C]1(C)C(C)=C(C)C(C)=C1C"),  # pentamethylcyclopentadienyl
    _extra(['CPh3', 'Ph3C', 'C(Ph)3'], "[C](c1ccccc1)(c1ccccc1)c1ccccc1"),  # trityl (triphenylmethyl)
    _extra(['crotyl', 'Crotyl', 'CH2CHCHCH3', 'CH2CHCHMe', 'MeCHCHCH2', 'CH2CH=CHCH3'], "[CH2]C=CC"),  # crotyl (but-2-enyl)
    _extra(['CT3', 'T3C'], "[C]([3H])([3H])[3H]"),  # tritritiomethyl
    _extra(['cyBu', 'cyclo-Bu', 'cyclo-C4H7', 'cycloBu'], "[CH]1CCC1"),  # cyclobutyl
    _extra(['cyclo-Pr', 'cyclo-C3H5', 'cycloPr'], "[CH]1CC1"),  # cyclopropyl
    _extra(['cyHex', 'cHx', 'c-Hx', 'Chx', 'cyclo-Hex', 'cyclo-C6H11', 'cycloHex', 'H11C6'], "[CH]1CCCCC1"),  # cyclohexyl
    _extra(['Cyp', 'cyPent', 'cPen', 'c-Pen', 'cyPen', 'cPe', 'c-Pe', 'cyclo-Pent', 'cyclo-C5H9', 'cycloPent'], "[CH]1CCCC1"),  # cyclopentyl
    _extra(['D'], "[2H]"),  # deuterium
    _extra(['D2O'], "[2H]O[2H]", 0),  # deuterium oxide
    _extra(['Dec', 'n-Dec', 'nDec', 'C10H21', 'n-C10H21', 'nC10H21', 'H21C10', '(CH2)9CH3'], "[CH2]CCCCCCCCC"),  # n-decyl
    _extra(['DMSO-d6', 'd6-DMSO'], "[2H]C([2H])([2H])S(=O)C([2H])([2H])[2H]", 0),  # dimethyl sulfoxide-d6
    _extra(['ethynyl', 'Ethynyl', 'CCH', 'HCC'], "[C]#C"),  # ethynyl
    _extra(['farnesyl', 'Farnesyl'], "[CH2]/C=C(\\C)CC/C=C(\\C)CCC=C(C)C"),  # farnesyl
    _extra(['geranyl', 'Geranyl'], "[CH2]/C=C(\\C)CCC=C(C)C"),  # geranyl
    _extra(['H2C'], "[CH2]", 2),  # methylene
    _extra(['H2CCH2'], "[CH2][CH2]", 2),  # ethane-1,2-diyl
    _extra(['HC'], "[CH]", 3),  # methine
    _extra(['homoallyl', 'Homoallyl', 'CH2CH2CHCH2', 'CH2CH2Vin', 'CH2CH2Vi', 'H2CCHCH2CH2', 'CH2CH2CH=CH2'], "[CH2]CC=C"),  # homoallyl (but-3-enyl)
    _extra(['homopropargyl', 'Homopropargyl', 'CH2CH2CCH', 'HCCCH2CH2'], "[CH2]CC#C"),  # homopropargyl (but-3-ynyl)
    _extra(['Hx', 'n-Hx', 'nHx', 'n-C6H13', 'nC6H13', 'H13C6', '(CH2)5CH3'], "[CH2]CCCCC"),  # n-hexyl
    _extra(['iAm', 'i-Am', 'iso-Am', 'isoAm', 'iPent', 'i-Pent', 'iso-Pent', 'isoPent', 'i-C5H11', 'iC5H11', 'iso-C5H11', 'CH2CH2CHMe2', 'Me2CHCH2CH2', 'CH2CH2CH(CH3)2', '(CH3)2CHCH2CH2', '(CH2)2CHMe2', '(CH2)2CH(CH3)2', 'CH2CH2iPr', 'iPrCH2CH2', 'iPr(CH2)2', '(CH2)2iPr'], "[CH2]CC(C)C"),  # isoamyl (isopentyl, 3-methylbutyl)
    _extra(['iHex', 'i-Hex', 'iso-Hex', 'CH2CH2CH2CHMe2', '(CH2)3CHMe2', '(CH2)3CH(CH3)2', 'Me2CH(CH2)3', '(CH2)3iPr', 'iPr(CH2)3'], "[CH2]CCC(C)C"),  # isohexyl (4-methylpentyl)
    _extra(['isobutenyl', 'Isobutenyl', 'CHCMe2', 'CHC(CH3)2', 'Me2CCH', 'CH=CMe2'], "[CH]=C(C)C"),  # isobutenyl (2-methylprop-1-enyl)
    _extra(['isoPr', 'iso-Pr', '2-Pr', 'i-C3H7', 'iC3H7', 'iso-C3H7', 'CHMe2', 'Me2CH', 'CH(CH3)2', '(CH3)2CH', '(H3C)2CH', 'CH(Me)2'], "[CH](C)C"),  # isopropyl
    _extra(['isopropenyl', 'Isopropenyl', 'C(Me)CH2', 'C(CH3)CH2', 'C(Me)=CH2', 'C(CH3)=CH2'], "[C](C)=C"),  # isopropenyl
    _extra(['methallyl', 'Methallyl', '2-methylallyl', 'CH2C(CH3)CH2', 'CH2C(Me)CH2', 'H2CC(Me)CH2'], "[CH2]C(C)=C"),  # methallyl (2-methylallyl)
    _extra(['N(CD3)2', '(CD3)2N', '(D3C)2N'], "[N](C([2H])([2H])[2H])C([2H])([2H])[2H]"),  # bis(trideuteromethyl)amino
    _extra(['n-C7H15', 'nC7H15', 'H15C7', '(CH2)6CH3'], "[CH2]CCCCCC"),  # n-heptyl
    _extra(['n-C8H17', 'nC8H17', 'H17C8', '(CH2)7CH3'], "[CH2]CCCCCCC"),  # n-octyl
    _extra(['n-Non', 'C9H19', 'n-C9H19', 'nC9H19', 'H19C9', '(CH2)8CH3'], "[CH2]CCCCCCCC"),  # n-nonyl
    _extra(['n-Pen', 'nPen', 'n-C5H11', 'nC5H11', 'H11C5', '(CH2)4CH3', 'CH2CH2CH2CH2CH3', 'n-Am', 'nAm'], "[CH2]CCCC"),  # n-pentyl (amyl)
    _extra(['ND'], "[N][2H]", 2),  # deuterated imino (ND)
    _extra(['ND2', 'D2N'], "[N]([2H])[2H]"),  # dideuteroamino
    _extra(['ND3', 'ND3+', 'D3N+'], "[N+]([2H])([2H])[2H]"),  # trideuteroammonio
    _extra(['neoPent', 'neo-Pent', 'neo-C5H11', 'neoC5H11', 'CH2tBu', 'tBuCH2', 'tBuH2C', 'CH2CMe3', 'Me3CCH2', 'CH2C(CH3)3', '(CH3)3CCH2', '(H3C)3CCH2'], "[CH2]C(C)(C)C"),  # neopentyl
    _extra(['neryl', 'Neryl'], "[CH2]/C=C(/C)CCC=C(C)C"),  # neryl
    _extra(['NHCD3', 'CD3NH', 'CD3HN', 'D3CNH', 'D3CHN'], "[NH]C([2H])([2H])[2H]"),  # trideuteromethylamino
    _extra(['NHD', 'NDH'], "[NH][2H]"),  # monodeuteroamino
    _extra(['OD', 'DO'], "[O][2H]"),  # deuteroxy (OD)
    _extra(['oleyl', 'Oleyl'], "[CH2]CCCCCCC/C=C\\CCCCCCCC"),  # oleyl ((Z)-octadec-9-en-1-yl)
    _extra(['phenylethynyl', 'Phenylethynyl', 'PhCC', 'CCPh'], "[C]#Cc1ccccc1"),  # phenylethynyl
    _extra(['prenyl', 'Prenyl', '3,3-dimethylallyl', 'CH2CHCMe2', 'CH2CHC(CH3)2', 'Me2CCHCH2', 'CH2CH=CMe2'], "[CH2]C=C(C)C"),  # prenyl (3-methylbut-2-enyl)
    _extra(['Prn', 'Pr-n', 'n-C3H7', 'nC3H7', 'H7C3', 'CH2CH2CH3', '(CH2)2CH3'], "[CH2]CC"),  # n-propyl
    _extra(['propargyl', 'Propargyl', 'CH2CCH', 'HCCCH2', 'HCCH2C'], "[CH2]C#C"),  # propargyl (prop-2-ynyl)
    _extra(['SCD3', 'CD3S', 'D3CS'], "[S]C([2H])([2H])[2H]"),  # trideuteromethylthio
    _extra(['SD', 'DS'], "[S][2H]"),  # deuterothio (SD)
    _extra(['styryl', 'Styryl', 'CHCHPh', 'PhCHCH', 'CH=CHPh', 'PhCH=CH'], "[CH]=Cc1ccccc1"),  # styryl (2-phenylethenyl)
    _extra(['T'], "[3H]"),  # tritium
    _extra(['tAm', 't-Am', 'tert-Am', 'Amt', 'tPent', 't-Pent', 'tert-Pent', 't-C5H11', 'tC5H11', 'tert-C5H11', 'CMe2Et', 'EtMe2C', 'C(CH3)2CH2CH3', 'C(CH3)2Et', 'C(Me)2Et'], "[C](C)(C)CC"),  # tert-amyl (tert-pentyl, 1,1-dimethylpropyl)
    _extra(['tertBu', 'tbu', 't-C4H9', 'tC4H9', 'tert-C4H9', 'CMe3', 'Me3C', 'C(CH3)3', '(CH3)3C', '(H3C)3C', 'C(Me)3'], "[C](C)(C)C"),  # tert-butyl
    _extra(['TES-ethynyl', 'CCTES', 'TESCC', 'CCSiEt3', 'Et3SiCC'], "[C]#C[Si](CC)(CC)CC"),  # (triethylsilyl)ethynyl
    _extra(['Thx', 'Thex', 'CMe2CHMe2', 'CMe2iPr', 'iPrMe2C', 'C(CH3)2CH(CH3)2'], "[C](C)(C)C(C)C"),  # thexyl (1,1,2-trimethylpropyl)
    _extra(['TIPS-ethynyl', 'CCTIPS', 'TIPSCC', 'CCSi(iPr)3', '(iPr)3SiCC', 'iPr3SiCC'], "[C]#C[Si](C(C)C)(C(C)C)C(C)C"),  # (triisopropylsilyl)ethynyl
    _extra(['TMS-ethynyl', 'CCTMS', 'TMSCC', 'CCSiMe3', 'Me3SiCC'], "[C]#C[Si](C)(C)C"),  # (trimethylsilyl)ethynyl
    _extra(['tOct', 't-Oct', 'tert-Oct', 'CMe2CH2tBu', 'CMe2CH2CMe3', 'C(CH3)2CH2C(CH3)3', 'tBuCH2CMe2', 'tBuCH2C(CH3)2'], "[C](C)(C)CC(C)(C)C"),  # tert-octyl (1,1,3,3-tetramethylbutyl)
    _extra(['Vin', 'Vi', 'vinyl', 'Vinyl', 'CHCH2', 'H2CCH', 'CH=CH2', 'H2C=CH'], "[CH]=C"),  # vinyl
    # linkers/bioconjugation
    _extra(['(CH2CH2O)1', 'CH2CH2O'], "[CH2]C[O]", 2),  # -(CH2CH2O)1- (left CH2, right O)
    _extra(['(CH2CH2O)2'], "[CH2]COCC[O]", 2),  # -(CH2CH2O)2- (left CH2, right O)
    _extra(['(CH2CH2O)3'], "[CH2]COCCOCC[O]", 2),  # -(CH2CH2O)3- (left CH2, right O)
    _extra(['(CH2CH2O)4'], "[CH2]COCCOCCOCC[O]", 2),  # -(CH2CH2O)4- (left CH2, right O)
    _extra(['(CH2CH2O)5'], "[CH2]COCCOCCOCCOCC[O]", 2),  # -(CH2CH2O)5- (left CH2, right O)
    _extra(['(CH2CH2O)6'], "[CH2]COCCOCCOCCOCCOCC[O]", 2),  # -(CH2CH2O)6- (left CH2, right O)
    _extra(['(OCH2CH2)1', 'OCH2CH2'], "[O]C[CH2]", 2),  # -(OCH2CH2)1- (left O, right CH2)
    _extra(['(OCH2CH2)2'], "[O]CCOC[CH2]", 2),  # -(OCH2CH2)2- (left O, right CH2)
    _extra(['(OCH2CH2)3'], "[O]CCOCCOC[CH2]", 2),  # -(OCH2CH2)3- (left O, right CH2)
    _extra(['(OCH2CH2)4'], "[O]CCOCCOCCOC[CH2]", 2),  # -(OCH2CH2)4- (left O, right CH2)
    _extra(['(OCH2CH2)5'], "[O]CCOCCOCCOCCOC[CH2]", 2),  # -(OCH2CH2)5- (left O, right CH2)
    _extra(['(OCH2CH2)6'], "[O]CCOCCOCCOCCOCCOC[CH2]", 2),  # -(OCH2CH2)6- (left O, right CH2)
    _extra(['1-Pyrenyl', 'Pyrenyl', '1-pyrenyl', 'pyrenyl', 'Pyrene', 'pyrene', '1-Pyrene'], "[c]1ccc2ccc3cccc4ccc1c2c34"),  # pyren-1-yl
    _extra(['AEEA', 'O2Oc', 'Aeea'], "[NH]CCOCCOC[C]=O", 2),  # 8-amino-3,6-dioxaoctanoyl residue -NH-CH2CH2-O-CH2CH2-O-CH2-C(=O)-
    _extra(['Aoa', 'AOA'], "[C](=O)CON"),  # aminooxyacetyl (free ONH2), via C=O
    _extra(['BCN'], "[C](=O)OCC1C2CCC#CCCC12"),  # BCN carbamate: (bicyclo[6.1.0]non-4-yn-9-yl)methoxycarbonyl, via C=O
    _extra(['Biotin', 'biotin', 'Biotinyl', 'biotinyl', 'D-Biotin', 'd-Biotin'], "[C](=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"),  # D-biotinyl (3aS,4S,6aR), via the side-chain C=O
    _extra(['C10H20'], "[CH2]CCCCCCCC[CH2]", 2),  # 10-carbon linear alkylene -(CH2)10-
    _extra(['C11H22'], "[CH2]CCCCCCCCC[CH2]", 2),  # 11-carbon linear alkylene -(CH2)11-
    _extra(['C12H24'], "[CH2]CCCCCCCCCC[CH2]", 2),  # 12-carbon linear alkylene -(CH2)12-
    _extra(['C2H4'], "[CH2][CH2]", 2),  # 2-carbon linear alkylene -(CH2)2-
    _extra(['C3H6'], "[CH2]C[CH2]", 2),  # 3-carbon linear alkylene -(CH2)3-
    _extra(['C4H8'], "[CH2]CC[CH2]", 2),  # 4-carbon linear alkylene -(CH2)4-
    _extra(['C5H10'], "[CH2]CCC[CH2]", 2),  # 5-carbon linear alkylene -(CH2)5-
    _extra(['C6H12'], "[CH2]CCCC[CH2]", 2),  # 6-carbon linear alkylene -(CH2)6-
    _extra(['C7H14'], "[CH2]CCCCC[CH2]", 2),  # 7-carbon linear alkylene -(CH2)7-
    _extra(['C8H16'], "[CH2]CCCCCC[CH2]", 2),  # 8-carbon linear alkylene -(CH2)8-
    _extra(['C9H18'], "[CH2]CCCCCCC[CH2]", 2),  # 9-carbon linear alkylene -(CH2)9-
    _extra(['CF2', 'F2C'], "[C](F)F", 2),  # difluoromethylene (in-line)
    _extra(['CH2CH2NH'], "[CH2]C[NH]", 2),  # -CH2CH2-NH- (left CH2, right N)
    _extra(['CH2NH'], "[CH2][NH]", 2),  # -CH2-NH- (left CH2, right N)
    _extra(['CH2O'], "[CH2][O]", 2),  # -CH2-O- (left CH2, right O)
    _extra(['CH2S'], "[CH2][S]", 2),  # -CH2-S- (left CH2, right S)
    _extra(['CO2K', 'KO2C', 'COOK', 'KOOC'], "[C](=O)[O-].[K+]"),  # potassium carboxylate
    _extra(['CO2Li', 'LiO2C', 'COOLi', 'LiOOC'], "[C](=O)[O-].[Li+]"),  # lithium carboxylate
    _extra(['CO2Na', 'NaO2C', 'COONa', 'NaOOC', 'CO2-Na+'], "[C](=O)[O-].[Na+]"),  # sodium carboxylate
    _extra(['CO2NH4', 'H4NO2C', 'COONH4'], "[C](=O)[O-].[NH4+]"),  # ammonium carboxylate
    _extra(['COCH2Br', 'BrCH2CO', 'C(O)CH2Br', 'BrH2CCO'], "[C](=O)CBr"),  # bromoacetyl, via C=O
    _extra(['COCH2CH2CO', 'CO(CH2)2CO'], "[C](=O)CC[C]=O", 2),  # succinyl link -C(=O)CH2CH2C(=O)-
    _extra(['COCH2Cl', 'ClCH2CO', 'C(O)CH2Cl', 'ClH2CCO'], "[C](=O)CCl"),  # chloroacetyl, via C=O
    _extra(['COCH2I', 'ICH2CO', 'C(O)CH2I', 'IH2CCO'], "[C](=O)CI"),  # iodoacetyl, via C=O
    _extra(['CONHNH2', 'H2NNHCO', 'H2NHNOC', 'C(O)NHNH2', 'H2NNHC(O)'], "[C](=O)NN"),  # carbohydrazide -C(=O)-NH-NH2
    _extra(['DBCO', 'DIBAC', 'ADIBO'], "[C](=O)CCC(=O)N1Cc2ccccc2C#Cc2ccccc21"),  # DBCO acyl: 4-(11,12-didehydrodibenzo[b,f]azocin-5(6H)-yl)-4-oxobutanoyl, via C=O
    _extra(['Desthiobiotin', 'desthiobiotin', 'Desthiobiotinyl', 'DTB', 'Dethiobiotin'], "[C](=O)CCCCC[C@@H]1NC(=O)N[C@@H]1C"),  # desthiobiotinyl (4R,5S), via the side-chain C=O
    _extra(['DIBO'], "[C](=O)OC1Cc2ccccc2C#Cc2ccccc21"),  # DIBO carbamate: 11,12-didehydro-5,6-dihydrodibenzo[a,e]cycloocten-5-yl-oxycarbonyl, via C=O
    _extra(['HNCO', 'HNC(O)'], "[NH][C]=O", 2),  # amide link -NH-C(=O)- (left N, right C)
    _extra(['HNCONH', 'HNC(O)NH'], "[NH]C(=O)[NH]", 2),  # urea link -NH-C(=O)-NH-
    _extra(['HS'], "[SH]"),  # thiol / mercapto
    _extra(['Mal', 'Maleimide', 'maleimide'], "[N]1C(=O)C=CC1=O"),  # maleimid-1-yl, via N
    _extra(['mc', 'MC'], "[C](=O)CCCCCN1C(=O)C=CC1=O"),  # 6-maleimidohexanoyl (maleimidocaproyl), via C=O
    _extra(['MCC', 'mcc', 'Mcc'], "[C](=O)C1CCC(CN2C(=O)C=CC2=O)CC1"),  # 4-(maleimidomethyl)cyclohexane-1-carbonyl (from SMCC), via C=O
    _extra(['Me3N', 'NMe3'], "CN(C)C", 0),  # trimethylamine (molecule)
    _extra(['MeTz', 'Me-Tz', 'MTz'], "[c]1nnc(C)nn1"),  # 6-methyl-1,2,4,5-tetrazin-3-yl
    _extra(['mp', 'MP'], "[C](=O)CCN1C(=O)C=CC1=O"),  # 3-maleimidopropanoyl, via C=O
    _extra(['NEt3', 'Et3N'], "CCN(CC)CC", 0),  # triethylamine (molecule)
    _extra(['NHBiotin', 'BiotinHN', 'BiotinNH'], "[NH]C(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"),  # biotinamide -NH-C(=O)-biotin, via N
    _extra(['NHCH2'], "[NH][CH2]", 2),  # -NH-CH2- (left N, right CH2)
    _extra(['NHCH2CH2'], "[NH]C[CH2]", 2),  # -NH-CH2CH2- (left N, right CH2)
    _extra(['NHCH2CH2CH2NH', 'NH(CH2)3NH'], "[NH]CCC[NH]", 2),  # propylenediamine link -NH-(CH2)3-NH-
    _extra(['NHCH2CH2NH', 'NH(CH2)2NH'], "[NH]CC[NH]", 2),  # ethylenediamine link -NH-CH2CH2-NH-
    _extra(['NHCOCH2Br', 'BrCH2CONH', 'BrCH2COHN', 'NHC(O)CH2Br'], "[NH]C(=O)CBr"),  # bromoacetamido, via N
    _extra(['NHCOCH2I', 'ICH2CONH', 'ICH2COHN', 'NHC(O)CH2I'], "[NH]C(=O)CI"),  # iodoacetamido, via N
    _extra(['NHCOO', 'NHC(O)O', 'HNCOO', 'HNC(O)O'], "[NH]C(=O)[O]", 2),  # carbamate link -NH-C(=O)-O- (left N, right O)
    _extra(['NHCSNH', 'NHC(S)NH', 'HNCSNH', 'HNC(S)NH'], "[NH]C(=S)[NH]", 2),  # thiourea link -NH-C(=S)-NH-
    _extra(['NHNH', 'HNNH'], "[NH][NH]", 2),  # hydrazine link -NH-NH-
    _extra(['NHO2S', 'HNSO2'], "[NH][S](=O)=O", 2),  # sulfonamide link -NH-SO2- (left N, right S)
    _extra(['NHS'], "[O]N1C(=O)CCC1=O"),  # NHS ester = N-succinimidyloxy (label on a carbonyl)
    _extra(['NHS', 'HOSu', 'HONSu'], "ON1C(=O)CCC1=O", 0),  # N-hydroxysuccinimide (reagent molecule)
    _extra(['O2SNH'], "[S](=O)(=O)[NH]", 2),  # sulfonamide link -SO2-NH- (left S, right N)
    _extra(['OAt', 'AtO'], "[O]n1nnc2cccnc21"),  # 7-azabenzotriazol-1-yloxy (OAt active ester)
    _extra(['OBt', 'BtO'], "[O]n1nnc2ccccc21"),  # 1H-benzotriazol-1-yloxy (OBt active ester)
    _extra(['OCH2'], "[O][CH2]", 2),  # -O-CH2- (left O, right CH2)
    _extra(['OCH2CH2O', 'O(CH2)2O'], "[O]CC[O]", 2),  # ethylenedioxy -O-CH2CH2-O-
    _extra(['OCONH', 'OC(O)NH'], "[O]C(=O)[NH]", 2),  # carbamate link -O-C(=O)-NH- (left O, right N)
    _extra(['OCOO', 'OC(O)O', 'OCO2'], "[O]C(=O)[O]", 2),  # carbonate link -O-C(=O)-O-
    _extra(['OK', 'KO'], "[O][K]"),  # potassium oxide/alkoxide -O-K (written covalently)
    _extra(['OLi', 'LiO'], "[O][Li]"),  # lithium alkoxide -O-Li (written covalently)
    _extra(['ONa', 'NaO'], "[O][Na]"),  # sodium oxide/alkoxide -O-Na (written covalently)
    _extra(['ONH2', 'H2NO'], "[O]N"),  # aminooxy -O-NH2
    _extra(['ONp', 'NpO', 'OpNP', 'pNPO', 'ONP'], "[O]c1ccc([N+](=O)[O-])cc1"),  # 4-nitrophenoxy (active-ester O)
    _extra(['OPfp', 'PfpO', 'OC6F5', 'F5C6O'], "[O]c1c(F)c(F)c(F)c(F)c1F"),  # pentafluorophenoxy (Pfp ester O)
    _extra(['OPO32-', '2-O3PO', 'OPO3'], "[O]P(=O)([O-])[O-]"),  # phosphate monoester dianion -O-PO3(2-)
    _extra(['OPO3H2', 'H2O3PO', 'OPO(OH)2', '(HO)2OPO', 'OP(O)(OH)2', '(HO)2(O)PO'], "[O]P(=O)(O)O"),  # phosphate monoester -O-PO3H2
    _extra(['OPO3Na2', 'Na2O3PO'], "[O]P(=O)([O-])[O-].[Na+].[Na+]"),  # disodium phosphate monoester
    _extra(['OSO3-', '-O3SO', 'OSO3'], "[O]S(=O)(=O)[O-]"),  # sulfate monoester anion -O-SO3-
    _extra(['OSO3H', 'HO3SO', 'OSO2OH'], "[O]S(=O)(=O)O"),  # sulfate monoester -O-SO3H
    _extra(['OSO3Na', 'NaO3SO'], "[O]S(=O)(=O)[O-].[Na+]"),  # sodium sulfate monoester
    _extra(['OSTP', 'STPO'], "[O]c1c(F)c(F)c(S(=O)(=O)O)c(F)c1F"),  # 4-sulfo-2,3,5,6-tetrafluorophenoxy (STP ester O)
    _extra(['OSu', 'SuO', 'ONHS'], "[O]N1C(=O)CCC1=O"),  # N-succinimidyloxy (NHS ester O)
    _extra(['OSuc', 'SucO'], "[O]C(=O)CCC(=O)O"),  # succinyloxy -O-C(=O)CH2CH2COOH (hemisuccinate)
    _extra(['OTfp', 'TfpO'], "[O]c1c(F)c(F)cc(F)c1F"),  # 2,3,5,6-tetrafluorophenoxy (TFP ester O)
    _extra(['PAB', 'pAB'], "[NH]c1ccc([CH2])cc1", 2),  # p-aminobenzyl spacer -NH-C6H4-CH2- (N left, CH2 right)
    _extra(['PAB', 'pAB'], "[CH2]c1ccc(N)cc1"),  # 4-aminobenzyl (free NH2), via CH2
    _extra(['PABA', 'pABA'], "[NH]c1ccc([C]=O)cc1", 2),  # 4-aminobenzoyl residue -NH-C6H4-C(=O)- (N left, C=O right)
    _extra(['PABA', 'pABA'], "[C](=O)c1ccc(N)cc1"),  # 4-aminobenzoyl (free NH2), via C=O
    _extra(['PABC', 'pABC'], "[NH]c1ccc(CO[C]=O)cc1", 2),  # p-aminobenzyl carbamate self-immolative spacer -NH-C6H4-CH2-O-C(=O)- (N left, carbamate C right)
    _extra(['PEG1', 'PEG-1', 'OEG1'], "[CH2]C[O]", 2),  # PEG1 spacer -(CH2CH2O)1- (left CH2, right O)
    _extra(['PEG10', 'PEG-10', 'OEG10'], "[CH2]COCCOCCOCCOCCOCCOCCOCCOCCOCC[O]", 2),  # PEG10 spacer -(CH2CH2O)10- (left CH2, right O)
    _extra(['PEG11', 'PEG-11', 'OEG11'], "[CH2]COCCOCCOCCOCCOCCOCCOCCOCCOCCOCC[O]", 2),  # PEG11 spacer -(CH2CH2O)11- (left CH2, right O)
    _extra(['PEG12', 'PEG-12', 'OEG12'], "[CH2]COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC[O]", 2),  # PEG12 spacer -(CH2CH2O)12- (left CH2, right O)
    _extra(['PEG16', 'PEG-16', 'OEG16'], "[CH2]COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC[O]", 2),  # PEG16 spacer -(CH2CH2O)16- (left CH2, right O)
    _extra(['PEG2', 'PEG-2', 'OEG2'], "[CH2]COCC[O]", 2),  # PEG2 spacer -(CH2CH2O)2- (left CH2, right O)
    _extra(['PEG24', 'PEG-24', 'OEG24'], "[CH2]COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC[O]", 2),  # PEG24 spacer -(CH2CH2O)24- (left CH2, right O)
    _extra(['PEG3', 'PEG-3', 'OEG3'], "[CH2]COCCOCC[O]", 2),  # PEG3 spacer -(CH2CH2O)3- (left CH2, right O)
    _extra(['PEG4', 'PEG-4', 'OEG4'], "[CH2]COCCOCCOCC[O]", 2),  # PEG4 spacer -(CH2CH2O)4- (left CH2, right O)
    _extra(['PEG5', 'PEG-5', 'OEG5'], "[CH2]COCCOCCOCCOCC[O]", 2),  # PEG5 spacer -(CH2CH2O)5- (left CH2, right O)
    _extra(['PEG6', 'PEG-6', 'OEG6'], "[CH2]COCCOCCOCCOCCOCC[O]", 2),  # PEG6 spacer -(CH2CH2O)6- (left CH2, right O)
    _extra(['PEG7', 'PEG-7', 'OEG7'], "[CH2]COCCOCCOCCOCCOCCOCC[O]", 2),  # PEG7 spacer -(CH2CH2O)7- (left CH2, right O)
    _extra(['PEG8', 'PEG-8', 'OEG8'], "[CH2]COCCOCCOCCOCCOCCOCCOCC[O]", 2),  # PEG8 spacer -(CH2CH2O)8- (left CH2, right O)
    _extra(['PEG9', 'PEG-9', 'OEG9'], "[CH2]COCCOCCOCCOCCOCCOCCOCCOCC[O]", 2),  # PEG9 spacer -(CH2CH2O)9- (left CH2, right O)
    _extra(['Pfp', 'PFP'], "[c]1c(F)c(F)c(F)c(F)c1F"),  # pentafluorophenyl
    _extra(['pNA', 'p-NA', 'NHpNP', 'pNPHN', 'pNPNH'], "[NH]c1ccc([N+](=O)[O-])cc1"),  # 4-nitroanilide, via N
    _extra(['pNP', 'p-NP', '4-NP'], "[c]1ccc([N+](=O)[O-])cc1"),  # 4-nitrophenyl
    _extra(['PO32-', '2-O3P'], "[P](=O)([O-])[O-]"),  # phosphonate dianion -PO3(2-)
    _extra(['PO3H2', 'H2O3P', 'PO(OH)2', '(HO)2OP', 'P(O)(OH)2', '(HO)2P(O)'], "[P](=O)(O)O"),  # phosphonic acid -PO3H2
    _extra(['SCH2'], "[S][CH2]", 2),  # -S-CH2- (left S, right CH2)
    _extra(['SO3-', '-O3S', 'SO3', 'O3S'], "[S](=O)(=O)[O-]"),  # sulfonate anion -SO3-
    _extra(['SO3K', 'KO3S'], "[S](=O)(=O)[O-].[K+]"),  # potassium sulfonate
    _extra(['SO3Na', 'NaO3S', 'SO3-Na+', 'Na+-O3S'], "[S](=O)(=O)[O-].[Na+]"),  # sodium sulfonate
    _extra(['SO3NH4', 'H4NO3S'], "[S](=O)(=O)[O-].[NH4+]"),  # ammonium sulfonate
    _extra(['SSPy', 'PySS'], "[S]Sc1ccccn1"),  # 2-pyridyldithio -S-S-(2-pyridyl)
    _extra(['Su', 'Succinimidyl', 'succinimidyl'], "[N]1C(=O)CCC1=O"),  # succinimid-1-yl, via N
    _extra(['Sulfo-NHS', 'sulfo-NHS', 'sNHS', 'sulfoNHS', 'SulfoNHS'], "[O]N1C(=O)CC(S(=O)(=O)O)C1=O"),  # sulfo-NHS ester O: N-(3-sulfosuccinimidyl)oxy (acid form)
    _extra(['Sulfo-NHS', 'sulfo-NHS', 'sNHS'], "ON1C(=O)CC(S(=O)(=O)[O-])C1=O.[Na+]", 0),  # N-hydroxysulfosuccinimide sodium salt (reagent molecule)
    _extra(['TCO'], "[C](=O)OC1CC/C=C/CCC1"),  # TCO carbamate: (E)-cyclooct-4-en-1-yloxycarbonyl, via C=O
    _extra(['Tfp'], "[c]1c(F)c(F)cc(F)c1F"),  # 2,3,5,6-tetrafluorophenyl
    _extra(['TNP', 'Tnp'], "[c]1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"),  # 2,4,6-trinitrophenyl
    _extra(['Ttds', 'TTDS'], "[NH]CCCOCCOCCOCCCNC(=O)CC[C]=O", 2),  # Ttds residue: 4,7,10-trioxa-1,13-tridecanediamine succinamic acid -NH-...-NH-C(=O)CH2CH2-C(=O)-
    # o_protect
    _extra(['CH3O'], "[O]C"),  # methoxy (reversed formula)
    _extra(['DEIPS'], "[Si](CC)(CC)C(C)C"),  # diethylisopropylsilyl
    _extra(['DMIPS', 'IPDMS'], "[Si](C)(C)C(C)C"),  # dimethylisopropylsilyl (isopropyldimethylsilyl)
    _extra(['DMPS'], "[Si](C)(C)c1ccccc1"),  # dimethylphenylsilyl
    _extra(['DPMS', 'MDPS'], "[Si](C)(c1ccccc1)c1ccccc1"),  # diphenylmethylsilyl (methyldiphenylsilyl)
    _extra(['DTBMS'], "[Si](C)(C(C)(C)C)C(C)(C)C"),  # di-tert-butylmethylsilyl
    _extra(['DTBS'], "[Si](C(C)(C)C)C(C)(C)C", 2),  # di-tert-butylsilylene (in-line between two O atoms)
    _extra(['EE'], "[CH](C)OCC"),  # 1-ethoxyethyl
    _extra(['EOM'], "[CH2]OCC"),  # ethoxymethyl
    _extra(['Fm'], "[CH2]C1c2ccccc2-c2ccccc21"),  # 9-fluorenylmethyl
    _extra(['iBuO', 'i-BuO', 'OCH2CHMe2', 'OCH2CH(CH3)2'], "[O]CC(C)C"),  # isobutoxy
    _extra(['Lev'], "[C](=O)CCC(C)=O"),  # levulinoyl (4-oxopentanoyl)
    _extra(['MTHP'], "[C]1(OC)CCOCC1"),  # 4-methoxytetrahydropyran-4-yl
    _extra(['MTM'], "[CH2]SC"),  # methylthiomethyl
    _extra(['OAll', 'AllO', 'OAllyl', 'AllylO', 'OCH2CH=CH2', 'CH2=CHCH2O'], "[O]CC=C"),  # allyloxy
    _extra(['OAlloc', 'AllocO', 'OCO2All', 'OCO2Allyl', 'OC(O)OAllyl', 'OCO2CH2CH=CH2'], "[O]C(=O)OCC=C"),  # allyloxycarbonyloxy (Alloc carbonate)
    _extra(['OBoc', 'BocO', 'OCO2tBu', 'OC(O)OtBu', 'tBuOCO2', 'tBuO2CO', 'OCO2t-Bu', 'OCOOtBu', 'OC(=O)OtBu'], "[O]C(=O)OC(C)(C)C"),  # tert-butoxycarbonyloxy (Boc carbonate)
    _extra(['OBOM', 'BOMO', 'OBom', 'BomO', 'OCH2OBn', 'BnOCH2O', 'OCH2OCH2Ph', 'PhCH2OCH2O'], "[O]COCc1ccccc1"),  # benzyloxymethoxy
    _extra(['OBpin', 'OB(pin)', 'pinBO', 'OBPin'], "[O]B1OC(C)(C)C(C)(C)O1"),  # pinacolboryloxy (boric ester)
    _extra(['OBu', 'OnBu', 'n-BuO', 'nBuO', 'BuO', 'OC4H9', 'C4H9O', 'O(CH2)3CH3'], "[O]CCCC"),  # butoxy
    _extra(['OC(CH3)3', '(CH3)3CO', 'OCMe3', 'Me3CO', 'OBu-t'], "[O]C(C)(C)C"),  # tert-butoxy (formula spellings)
    _extra(['OC(Me)(OMe)O'], "[O]C(C)(OC)[O]", 2),  # 1-methoxyethylidenedioxy (orthoester), in-line
    _extra(['OC(Me)(Ph)O', 'OCMePhO'], "[O]C(C)(c1ccccc1)[O]", 2),  # 1-phenylethylidenedioxy, in-line
    _extra(['OC(Me)2O', 'OCMe2O', 'OC(CH3)2O', 'O(CMe2)O'], "[O]C(C)(C)[O]", 2),  # isopropylidenedioxy (acetonide), in-line
    _extra(['OC(O)CF3', 'OC(=O)CF3', 'F3CCO2', 'F3CCOO', 'CF3C(O)O'], "[O]C(=O)C(F)(F)F"),  # trifluoroacetoxy (formula spellings)
    _extra(['OC(O)Ph', 'OC(=O)Ph', 'PhC(O)O', 'OCOC6H5', 'C6H5CO2'], "[O]C(=O)c1ccccc1"),  # benzoyloxy (formula spellings)
    _extra(['OC(S)Im', 'OC(=S)Im', 'OC(S)imid', 'ImC(S)O'], "[O]C(=S)n1ccnc1"),  # imidazol-1-ylthiocarbonyloxy (thiocarbonylimidazolide)
    _extra(['OC(S)NHPh', 'OC(=S)NHPh', 'PhHNC(S)O'], "[O]C(=S)Nc1ccccc1"),  # phenylthiocarbamoyloxy
    _extra(['OC(S)NMe2', 'OC(=S)NMe2', 'Me2NC(S)O', 'Me2NC(=S)O', 'OCSNMe2'], "[O]C(=S)N(C)C"),  # dimethylthiocarbamoyloxy (Newman-Kwart O-thiocarbamate)
    _extra(['OC(S)OPh', 'OC(=S)OPh', 'PhOC(S)O', 'PhOC(=S)O', 'OCSOPh'], "[O]C(=S)Oc1ccccc1"),  # phenoxythiocarbonyloxy (Barton-McCombie)
    _extra(['OC(S)SBn', 'OC(=S)SBn', 'BnSC(S)O'], "[O]C(=S)SCc1ccccc1"),  # (benzylthio)thiocarbonyloxy
    _extra(['OC(S)SMe', 'OC(=S)SMe', 'OCS2Me', 'MeSC(S)O', 'MeSC(=S)O'], "[O]C(=S)SC"),  # (methylthio)thiocarbonyloxy (S-methyl xanthate)
    _extra(['OC10H21', 'C10H21O', 'H21C10O', 'ODec'], "[O]CCCCCCCCCC"),  # decyloxy
    _extra(['OC12H25', 'C12H25O', 'H25C12O', 'O(CH2)11CH3'], "[O]CCCCCCCCCCCC"),  # dodecyloxy
    _extra(['OC14H29', 'C14H29O', 'H29C14O'], "[O]CCCCCCCCCCCCCC"),  # tetradecyloxy
    _extra(['OC16H33', 'C16H33O', 'H33C16O'], "[O]CCCCCCCCCCCCCCCC"),  # hexadecyloxy
    _extra(['OC18H37', 'C18H37O', 'H37C18O'], "[O]CCCCCCCCCCCCCCCCCC"),  # octadecyloxy
    _extra(['OC2H5', 'C2H5O', 'H5C2O'], "[O]CC"),  # ethoxy (formula)
    _extra(['OC6H4Me', 'OC6H4CH3'], "[O]c1ccc(C)cc1"),  # 4-methylphenoxy (p-tolyloxy)
    _extra(['OC6H5', 'C6H5O', 'H5C6O'], "[O]c1ccccc1"),  # phenoxy (formula spellings)
    _extra(['OcBu', 'cBuO'], "[O]C1CCC1"),  # cyclobutoxy
    _extra(['OCbz', 'CbzO', 'OCO2Bn', 'OC(O)OBn', 'BnOCO2', 'BnO2CO', 'OCOOBn', 'OC(=O)OBn', 'OCO2CH2Ph'], "[O]C(=O)OCc1ccccc1"),  # benzyloxycarbonyloxy (Cbz carbonate)
    _extra(['OCH(CH3)2', '(CH3)2CHO', 'OCHMe2', 'Me2CHO', 'OPr-i'], "[O]C(C)C"),  # isopropoxy (formula spellings)
    _extra(['OCH2CCH', 'HCCCH2O', 'OPropargyl', 'PropargylO'], "[O]CC#C"),  # propargyloxy
    _extra(['OCH2CH2CH2NMe2', 'O(CH2)3NMe2', 'Me2N(CH2)3O'], "[O]CCCN(C)C"),  # 3-(dimethylamino)propoxy
    _extra(['OCH2CH2CH2OH', 'O(CH2)3OH', 'HO(CH2)3O', 'HOCH2CH2CH2O'], "[O]CCCO"),  # 3-hydroxypropoxy
    _extra(['OCH2CH2NEt2', 'Et2NCH2CH2O', 'O(CH2)2NEt2'], "[O]CCN(CC)CC"),  # 2-(diethylamino)ethoxy
    _extra(['OCH2CH2NH2', 'H2NCH2CH2O', 'O(CH2)2NH2', 'H2N(CH2)2O'], "[O]CCN"),  # 2-aminoethoxy
    _extra(['OCH2CH2NMe2', 'Me2NCH2CH2O', 'O(CH2)2NMe2', 'Me2N(CH2)2O'], "[O]CCN(C)C"),  # 2-(dimethylamino)ethoxy
    _extra(['OCH2CH2OCH2CH2OMe', 'O(CH2CH2O)2Me', 'MeO(CH2CH2O)2', 'MeOCH2CH2OCH2CH2O'], "[O]CCOCCOC"),  # 2-(2-methoxyethoxy)ethoxy
    _extra(['OCH2CH2OH', 'HOCH2CH2O', 'O(CH2)2OH', 'HO(CH2)2O'], "[O]CCO"),  # 2-hydroxyethoxy
    _extra(['OCH2CH2OMe', 'MeOCH2CH2O', 'O(CH2)2OMe', 'OCH2CH2OCH3', 'CH3OCH2CH2O', 'MeO(CH2)2O'], "[O]CCOC"),  # 2-methoxyethoxy
    _extra(['OCH2CH2Ph', 'PhCH2CH2O', 'O(CH2)2Ph', 'OCH2Bn'], "[O]CCc1ccccc1"),  # 2-phenylethoxy
    _extra(['OCH2CN', 'NCCH2O'], "[O]CC#N"),  # cyanomethoxy
    _extra(['OCH2CO2Et', 'EtO2CCH2O', 'OCH2COOEt'], "[O]CC(=O)OCC"),  # (ethoxycarbonyl)methoxy
    _extra(['OCH2CO2H', 'HO2CCH2O', 'OCH2COOH', 'HOOCCH2O'], "[O]CC(=O)O"),  # carboxymethoxy
    _extra(['OCH2CO2Me', 'MeO2CCH2O', 'OCH2COOMe', 'OCH2CO2CH3'], "[O]CC(=O)OC"),  # (methoxycarbonyl)methoxy
    _extra(['OCH2CO2tBu', 'tBuO2CCH2O', 'OCH2CO2t-Bu'], "[O]CC(=O)OC(C)(C)C"),  # (tert-butoxycarbonyl)methoxy
    _extra(['OCH2CONH2', 'H2NOCCH2O', 'H2NCOCH2O'], "[O]CC(N)=O"),  # carbamoylmethoxy
    _extra(['OCH2cPr', 'cPrCH2O'], "[O]CC1CC1"),  # cyclopropylmethoxy
    _extra(['OCH2Cy', 'CyCH2O', 'OCH2cHex'], "[O]CC1CCCCC1"),  # cyclohexylmethoxy
    _extra(['OCH2OCH2CH2OMe', 'MeOCH2CH2OCH2O', 'OCH2O(CH2)2OMe'], "[O]COCCOC"),  # 2-methoxyethoxymethoxy (MEM-O formula)
    _extra(['OCH2OCH2CH2SiMe3', 'OCH2OCH2CH2TMS', 'Me3SiCH2CH2OCH2O', 'TMSCH2CH2OCH2O', 'OCH2O(CH2)2TMS'], "[O]COCC[Si](C)(C)C"),  # 2-(trimethylsilyl)ethoxymethoxy (SEM-O formula)
    _extra(['OCH2OMe', 'MeOCH2O', 'OCH2OCH3', 'CH3OCH2O'], "[O]COC"),  # methoxymethoxy (MOM-O formula)
    _extra(['OCH2Ph', 'PhCH2O', 'OCH2C6H5', 'C6H5CH2O', 'OBzl', 'BzlO'], "[O]Cc1ccccc1"),  # benzyloxy (formula / peptide spellings)
    _extra(['OCH2tBu', 'tBuCH2O', 'OCH2CMe3', 'OCH2C(CH3)3', 'OneoPent'], "[O]CC(C)(C)C"),  # neopentyloxy
    _extra(['OCHO', 'HCO2', 'HCOO', 'OC(O)H', 'OC(=O)H'], "[O]C=O"),  # formyloxy
    _extra(['OCHPh2', 'Ph2CHO', 'OBzh', 'BzhO'], "[O]C(c1ccccc1)c1ccccc1"),  # benzhydryloxy (diphenylmethoxy)
    _extra(['OCHPhO', 'OCH(Ph)O'], "[O]C(c1ccccc1)[O]", 2),  # benzylidenedioxy, in-line
    _extra(['OClAc', 'ClAcO', 'OCOCH2Cl', 'ClCH2CO2', 'ClCH2COO', 'OC(O)CH2Cl', 'ClCH2C(O)O'], "[O]C(=O)CCl"),  # chloroacetoxy
    _extra(['OCO(CH2)2CO2H', 'OCOCH2CH2CO2H', 'OCOCH2CH2COOH', 'HO2CCH2CH2CO2'], "[O]C(=O)CCC(=O)O"),  # 3-carboxypropanoyloxy (hemisuccinate, formula spelling)
    _extra(['OCO2Et', 'OC(O)OEt', 'OC(=O)OEt', 'EtOCO2', 'EtO2CO', 'OCOOEt', 'OCO2C2H5', 'OEoc', 'EocO'], "[O]C(=O)OCC"),  # ethoxycarbonyloxy (ethyl carbonate)
    _extra(['OCO2Me', 'OC(O)OMe', 'OC(=O)OMe', 'MeOCO2', 'MeO2CO', 'OCOOMe', 'OCO2CH3', 'OC(O)OCH3', 'CH3OCO2', 'OCOOCH3', 'OMoc', 'MocO'], "[O]C(=O)OC"),  # methoxycarbonyloxy (methyl carbonate)
    _extra(['OCO2Ph', 'OC(O)OPh', 'OC(=O)OPh', 'PhOCO2', 'PhO2CO', 'OCOOPh'], "[O]C(=O)Oc1ccccc1"),  # phenoxycarbonyloxy (phenyl carbonate)
    _extra(['OCO2Su', 'OC(O)OSu', 'OCO2NHS', 'SuOCO2'], "[O]C(=O)ON1C(=O)CCC1=O"),  # succinimidyloxycarbonyloxy (activated carbonate)
    _extra(['OCOC(Me)=CH2', 'OC(O)C(Me)=CH2', 'OCOC(CH3)=CH2', 'OC(O)C(CH3)=CH2', 'CH2=C(Me)CO2', 'CH2=C(CH3)CO2'], "[O]C(=O)C(C)=C"),  # methacryloyloxy
    _extra(['OCOCCl3', 'Cl3CCO2', 'Cl3CCOO', 'OC(O)CCl3'], "[O]C(=O)C(Cl)(Cl)Cl"),  # trichloroacetoxy
    _extra(['OCOCH2CH2CO2Me', 'OCO(CH2)2CO2Me'], "[O]C(=O)CCC(=O)OC"),  # 3-(methoxycarbonyl)propanoyloxy
    _extra(['OCOCH2Ph', 'OCOBn', 'PhCH2CO2', 'BnCO2', 'PhCH2COO', 'OC(O)Bn', 'OC(O)CH2Ph'], "[O]C(=O)Cc1ccccc1"),  # phenylacetoxy
    _extra(['OCOCH=CH2', 'OC(O)CH=CH2', 'CH2=CHCO2', 'CH2=CHCOO'], "[O]C(=O)C=C"),  # acryloyloxy
    _extra(['OCOCHCl2', 'Cl2CHCO2', 'Cl2CHCOO', 'OC(O)CHCl2'], "[O]C(=O)C(Cl)Cl"),  # dichloroacetoxy
    _extra(['OCOEt', 'OC(O)Et', 'OC(=O)Et', 'EtCO2', 'EtCOO', 'OCOCH2CH3'], "[O]C(=O)CC"),  # propanoyloxy
    _extra(['OCOiPr', 'OC(O)iPr', 'OC(=O)iPr', 'iPrCO2', 'iPrCOO', 'OCOCHMe2', 'OCOCH(CH3)2', 'i-Pr(O)CO'], "[O]C(=O)C(C)C"),  # isobutyryloxy
    _extra(['OCOMe', 'OC(O)Me', 'OC(=O)Me', 'OC(O)CH3', 'OC(=O)CH3', 'MeC(O)O', 'H3CCO2', 'H3CCOO'], "[O]C(C)=O"),  # acetoxy (formula spellings)
    _extra(['OCONEt2', 'OC(O)NEt2', 'OC(=O)NEt2', 'Et2NCO2', 'Et2NOCO', 'Et2NC(O)O', 'OCON(Et)2', 'OC(O)N(Et)2'], "[O]C(=O)N(CC)CC"),  # diethylcarbamoyloxy (DoM directing carbamate)
    _extra(['OCONH2', 'H2NCO2', 'H2NOCO', 'OC(O)NH2', 'OC(=O)NH2', 'H2NC(O)O', 'H2NCOO'], "[O]C(N)=O"),  # carbamoyloxy
    _extra(['OCONHBn', 'OC(O)NHBn', 'BnHNCO2', 'BnNHCO2'], "[O]C(=O)NCc1ccccc1"),  # benzylcarbamoyloxy
    _extra(['OCONHEt', 'OC(O)NHEt', 'EtHNCO2', 'EtNHCO2'], "[O]C(=O)NCC"),  # ethylcarbamoyloxy
    _extra(['OCONHMe', 'OC(O)NHMe', 'OC(=O)NHMe', 'MeHNCO2', 'MeNHCO2', 'MeHNOCO', 'MeNHC(O)O', 'OCONHCH3'], "[O]C(=O)NC"),  # methylcarbamoyloxy
    _extra(['OCONHPh', 'OC(O)NHPh', 'OC(=O)NHPh', 'PhHNCO2', 'PhNHCO2', 'PhNHC(O)O', 'PhHNOCO'], "[O]C(=O)Nc1ccccc1"),  # phenylcarbamoyloxy
    _extra(['OCONHtBu', 'OC(O)NHtBu', 'tBuHNCO2', 'tBuNHCO2'], "[O]C(=O)NC(C)(C)C"),  # tert-butylcarbamoyloxy
    _extra(['OCONiPr2', 'OC(O)NiPr2', 'OC(=O)NiPr2', 'OCON(iPr)2', 'OC(O)N(iPr)2', 'iPr2NCO2', 'iPr2NC(O)O', 'iPr2NOCO', 'OCON(i-Pr)2', 'OC(O)N(i-Pr)2'], "[O]C(=O)N(C(C)C)C(C)C"),  # diisopropylcarbamoyloxy
    _extra(['OCONMe2', 'OC(O)NMe2', 'OC(=O)NMe2', 'Me2NCO2', 'Me2NOCO', 'Me2NC(O)O', 'OCON(CH3)2', 'OC(O)N(CH3)2'], "[O]C(=O)N(C)C"),  # dimethylcarbamoyloxy
    _extra(['OCOtBu', 'OC(O)tBu', 'OC(=O)tBu', 'tBuCO2', 'tBuCOO', 't-BuCO2', 'OCOC(CH3)3', '(CH3)3CCO2', 'OCOCMe3', 'Me3CCO2'], "[O]C(=O)C(C)(C)C"),  # pivaloyloxy (formula spellings)
    _extra(['OcPent', 'cPentO', 'OcPen'], "[O]C1CCCC1"),  # cyclopentyloxy
    _extra(['OCPh2O', 'OC(Ph)2O'], "[O]C(c1ccccc1)(c1ccccc1)[O]", 2),  # diphenylmethylenedioxy, in-line
    _extra(['OCPh3', 'Ph3CO', 'OC(Ph)3'], "[O]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # trityloxy (formula spellings)
    _extra(['OcPr', 'cPrO'], "[O]C1CC1"),  # cyclopropoxy
    _extra(['OCy', 'CyO', 'OcHex', 'OC6H11', 'C6H11O'], "[O]C1CCCCC1"),  # cyclohexyloxy
    _extra(['ODEIPS', 'DEIPSO'], "[O][Si](CC)(CC)C(C)C"),  # diethylisopropylsilyloxy
    _extra(['ODMB', 'DMBO'], "[O]Cc1ccc(OC)cc1OC"),  # 2,4-dimethoxybenzyloxy
    _extra(['ODMIPS', 'DMIPSO', 'OIPDMS', 'IPDMSO'], "[O][Si](C)(C)C(C)C"),  # dimethylisopropylsilyloxy
    _extra(['ODMPS', 'DMPSO'], "[O][Si](C)(C)c1ccccc1"),  # dimethylphenylsilyloxy
    _extra(['ODMTr', 'DMTrO', 'ODMT', 'DMTO'], "[O]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1"),  # 4,4'-dimethoxytrityloxy
    _extra(['ODPMS', 'DPMSO'], "[O][Si](C)(c1ccccc1)c1ccccc1"),  # diphenylmethylsilyloxy
    _extra(['ODTBMS', 'DTBMSO'], "[O][Si](C)(C(C)(C)C)C(C)(C)C"),  # di-tert-butylmethylsilyloxy
    _extra(['OEE', 'EEO', 'OCH(Me)OEt', 'OCH(OEt)Me', 'OCH(CH3)OEt', 'OCH(OEt)CH3'], "[O]C(C)OCC"),  # 1-ethoxyethoxy
    _extra(['OEOM', 'EOMO', 'OCH2OEt', 'EtOCH2O'], "[O]COCC"),  # ethoxymethoxy
    _extra(['OFm', 'FmO'], "[O]CC1c2ccccc2-c2ccccc21"),  # 9-fluorenylmethoxy
    _extra(['OFmoc', 'FmocO'], "[O]C(=O)OCC1c2ccccc2-c2ccccc21"),  # 9-fluorenylmethoxycarbonyloxy (Fmoc carbonate)
    _extra(['OHex', 'OnHex', 'OC6H13', 'C6H13O', 'H13C6O', 'O(CH2)5CH3'], "[O]CCCCCC"),  # hexyloxy
    _extra(['OLev', 'LevO'], "[O]C(=O)CCC(C)=O"),  # levulinoyloxy
    _extra(['OMes', 'MesO'], "[O]c1c(C)cc(C)cc1C"),  # 2,4,6-trimethylphenoxy (mesityloxy)
    _extra(['OMMTr', 'MMTrO', 'OMMT', 'MMTO'], "[O]C(c1ccc(OC)cc1)(c1ccccc1)c1ccccc1"),  # 4-methoxytrityloxy
    _extra(['OMPM', 'MPMO'], "[O]Cc1ccc(OC)cc1"),  # 4-methoxybenzyloxy (MPM spelling)
    _extra(['OMTHP', 'MTHPO'], "[O]C1(OC)CCOCC1"),  # 4-methoxytetrahydropyran-4-yloxy
    _extra(['OMTM', 'MTMO', 'OCH2SMe', 'MeSCH2O', 'OCH2SCH3'], "[O]CSC"),  # methylthiomethoxy
    _extra(['ONap', 'NapO'], "[O]Cc1ccc2ccccc2c1"),  # 2-naphthylmethoxy (Nap ether)
    _extra(['OOct', 'OnOct', 'OC8H17', 'C8H17O', 'H17C8O', 'O(CH2)7CH3'], "[O]CCCCCCCC"),  # octyloxy
    _extra(['OP(O)(NEt2)2', 'OPO(NEt2)2', '(Et2N)2P(O)O'], "[O]P(=O)(N(CC)CC)N(CC)CC"),  # bis(diethylamino)phosphoryloxy
    _extra(['OP(O)(NMe2)2', 'OPO(NMe2)2', '(Me2N)2P(O)O'], "[O]P(=O)(N(C)C)N(C)C"),  # bis(dimethylamino)phosphoryloxy (tetramethylphosphorodiamidate)
    _extra(['OP(O)(OBn)2', 'OPO(OBn)2', 'OPO3Bn2', '(BnO)2P(O)O', '(BnO)2OPO', 'OP(=O)(OBn)2'], "[O]P(=O)(OCc1ccccc1)OCc1ccccc1"),  # dibenzyloxyphosphoryloxy (dibenzyl phosphate)
    _extra(['OP(O)(OEt)2', 'OPO(OEt)2', 'OPO3Et2', '(EtO)2P(O)O', '(EtO)2OPO', '(EtO)2(O)PO', 'OP(=O)(OEt)2', 'OP(O)(OC2H5)2', 'Et2O3PO'], "[O]P(=O)(OCC)OCC"),  # diethoxyphosphoryloxy (diethyl phosphate)
    _extra(['OP(O)(OEt)Ph', 'OP(O)Ph(OEt)'], "[O]P(=O)(OCC)c1ccccc1"),  # ethoxy(phenyl)phosphoryloxy
    _extra(['OP(O)(OH)OMe', 'OPO(OH)OMe', 'OPO3HMe'], "[O]P(=O)(O)OC"),  # methyl hydrogen phosphate ester
    _extra(['OP(O)(OiPr)2', 'OPO(OiPr)2', '(iPrO)2P(O)O'], "[O]P(=O)(OC(C)C)OC(C)C"),  # diisopropoxyphosphoryloxy
    _extra(['OP(O)(OMe)2', 'OPO(OMe)2', 'OPO3Me2', '(MeO)2P(O)O', '(MeO)2OPO', '(MeO)2(O)PO', 'OP(=O)(OMe)2', 'Me2O3PO'], "[O]P(=O)(OC)OC"),  # dimethoxyphosphoryloxy (dimethyl phosphate)
    _extra(['OP(O)(OPh)2', 'OPO(OPh)2', 'OPO3Ph2', '(PhO)2P(O)O', '(PhO)2OPO', 'OP(=O)(OPh)2'], "[O]P(=O)(Oc1ccccc1)Oc1ccccc1"),  # diphenoxyphosphoryloxy (diphenyl phosphate)
    _extra(['OP(O)(OtBu)2', 'OPO(OtBu)2', 'OPO3tBu2', '(tBuO)2P(O)O', '(tBuO)2OPO', 'OP(=O)(OtBu)2'], "[O]P(=O)(OC(C)(C)C)OC(C)(C)C"),  # di-tert-butoxyphosphoryloxy (di-tert-butyl phosphate)
    _extra(['OP(O)Me2', 'Me2P(O)O', 'OP(=O)Me2'], "[O]P(=O)(C)C"),  # dimethylphosphinoyloxy
    _extra(['OP(O)Ph2', 'OPOPh2', 'Ph2P(O)O', 'Ph2(O)PO', 'OP(=O)Ph2', 'Ph2OPO'], "[O]P(=O)(c1ccccc1)c1ccccc1"),  # diphenylphosphinoyloxy (diphenylphosphinate)
    _extra(['OP(OEt)2', 'OP(OC2H5)2'], "[O]P(OCC)OCC"),  # diethoxyphosphanyloxy (diethyl phosphite)
    _extra(['OP(OMe)2'], "[O]P(OC)OC"),  # dimethoxyphosphanyloxy (dimethyl phosphite)
    _extra(['OP(OPh)2'], "[O]P(Oc1ccccc1)Oc1ccccc1"),  # diphenoxyphosphanyloxy (diphenyl phosphite)
    _extra(['OPac', 'PacO'], "[O]CC(=O)c1ccccc1"),  # phenacyloxy
    _extra(['OPent', 'OnPent', 'OC5H11', 'C5H11O', 'H11C5O'], "[O]CCCCC"),  # pentyloxy
    _extra(['OPMBM', 'PMBMO'], "[O]COCc1ccc(OC)cc1"),  # 4-methoxybenzyloxymethoxy
    _extra(['OPO3HNa', 'NaHO3PO'], "[O]P(=O)(O)[O-].[Na+]"),  # monosodium phosphate ester
    _extra(['OPP', 'PPO', 'OP2O6H3'], "[O]P(=O)(O)OP(=O)(O)O"),  # diphosphate (pyrophosphate) ester oxygen, drawn as the free acid
    _extra(['OPr', 'OnPr', 'n-PrO', 'nPrO', 'PrO', 'OC3H7', 'C3H7O', 'OCH2CH2CH3'], "[O]CCC"),  # propoxy
    _extra(['OPx', 'PxO'], "[O]C1(c2ccccc2)c2ccccc2Oc2ccccc21"),  # 9-phenylxanthen-9-yloxy
    _extra(['OsBu', 'sBuO', 'sec-BuO', 'OCH(Me)Et'], "[O]C(C)CC"),  # sec-butoxy
    _extra(['OSi(iPr)3', 'OSiiPr3', 'iPr3SiO', '(iPr)3SiO', 'OSi(i-Pr)3', '(i-Pr)3SiO'], "[O][Si](C(C)C)(C(C)C)C(C)C"),  # triisopropylsilyloxy (formula spellings)
    _extra(['OSi(tBu)2O', 'OSitBu2O', 'OSi(t-Bu)2O'], "[O][Si](C(C)(C)C)(C(C)(C)C)[O]", 2),  # di-tert-butylsilylene diol protection (in-line)
    _extra(['OSiEt3', 'Et3SiO'], "[O][Si](CC)(CC)CC"),  # triethylsilyloxy (formula spellings)
    _extra(['OSiMe2Ph', 'PhMe2SiO', 'OSiPhMe2'], "[O][Si](C)(C)c1ccccc1"),  # dimethylphenylsilyloxy (formula spellings)
    _extra(['OSiMe2tBu', 'OSitBuMe2', 'tBuMe2SiO', 'OSi(tBu)Me2', 'OSiMe2(tBu)', 'OSiMe2t-Bu', 't-BuMe2SiO'], "[O][Si](C)(C)C(C)(C)C"),  # tert-butyldimethylsilyloxy (formula spellings)
    _extra(['OSiMe3', 'Me3SiO', 'OSi(CH3)3', '(CH3)3SiO', '(H3C)3SiO'], "[O][Si](C)(C)C"),  # trimethylsilyloxy (formula spellings)
    _extra(['OSiMePh2', 'Ph2MeSiO', 'OSiPh2Me'], "[O][Si](C)(c1ccccc1)c1ccccc1"),  # methyldiphenylsilyloxy (formula spellings)
    _extra(['OSiPh2tBu', 'OSitBuPh2', 'tBuPh2SiO', 'OSi(tBu)Ph2', 't-BuPh2SiO'], "[O][Si](c1ccccc1)(c1ccccc1)C(C)(C)C"),  # tert-butyldiphenylsilyloxy (formula spellings)
    _extra(['OSiPh3', 'Ph3SiO'], "[O][Si](c1ccccc1)(c1ccccc1)c1ccccc1"),  # triphenylsilyloxy
    _extra(['OSO2C4F9', 'C4F9SO2O'], "[O]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),  # nonafluorobutanesulfonyloxy (nonaflate, formula spelling)
    _extra(['OSO2CF3', 'CF3SO2O', 'F3CSO2O', 'F3CO2SO', 'CF3O2SO', 'OS(O)2CF3'], "[O]S(=O)(=O)C(F)(F)F"),  # trifluoromethanesulfonyloxy (triflyloxy, formula spellings)
    _extra(['OSO2CH2CF3', 'CF3CH2SO2O', 'OTres', 'TresO'], "[O]S(=O)(=O)CC(F)(F)F"),  # 2,2,2-trifluoroethanesulfonyloxy (tresylate)
    _extra(['OSO2Cl', 'ClSO2O'], "[O]S(Cl)(=O)=O"),  # chlorosulfonyloxy
    _extra(['OSO2Et', 'EtSO2O'], "[O]S(=O)(=O)CC"),  # ethanesulfonyloxy
    _extra(['OSO2F', 'FSO2O', 'FO2SO'], "[O]S(F)(=O)=O"),  # fluorosulfonyloxy (fluorosulfate, SuFEx)
    _extra(['OSO2Me', 'MeSO2O', 'MeO2SO', 'OSO2CH3', 'CH3SO2O', 'H3CO2SO', 'OS(O)2Me'], "[O]S(C)(=O)=O"),  # methanesulfonyloxy (mesyloxy, formula spellings)
    _extra(['OSO2Mes', 'MesSO2O', 'OMts', 'MtsO'], "[O]S(=O)(=O)c1c(C)cc(C)cc1C"),  # mesitylenesulfonyloxy
    _extra(['OSO2NEt2', 'Et2NSO2O'], "[O]S(=O)(=O)N(CC)CC"),  # diethylsulfamoyloxy
    _extra(['OSO2NH2', 'H2NSO2O', 'H2NO2SO', 'OS(O)2NH2'], "[O]S(N)(=O)=O"),  # sulfamoyloxy (sulfamate)
    _extra(['OSO2NMe2', 'Me2NSO2O', 'Me2NO2SO'], "[O]S(=O)(=O)N(C)C"),  # dimethylsulfamoyloxy
    _extra(['OSO2OMe', 'MeOSO2O', 'OSO3Me'], "[O]S(=O)(=O)OC"),  # methoxysulfonyloxy (methyl sulfate ester)
    _extra(['OSO2Ph', 'PhSO2O', 'PhO2SO', 'OSO2C6H5'], "[O]S(=O)(=O)c1ccccc1"),  # benzenesulfonyloxy (besyloxy)
    _extra(['OSO2Tol', 'OSO2pTol', 'OSO2p-Tol', 'TolSO2O', 'pTolSO2O', 'OSO2C6H4Me', 'OSO2C6H4CH3', 'MeC6H4SO2O'], "[O]S(=O)(=O)c1ccc(C)cc1"),  # p-toluenesulfonyloxy (tosyloxy, formula spellings)
    _extra(['OTce', 'TceO', 'OCH2CCl3', 'Cl3CCH2O'], "[O]CC(Cl)(Cl)Cl"),  # 2,2,2-trichloroethoxy
    _extra(['OTDS', 'TDSO'], "[O][Si](C)(C)C(C)(C)C(C)C"),  # thexyldimethylsilyloxy
    _extra(['OTeoc', 'TeocO', 'OCO2CH2CH2TMS', 'OCO2CH2CH2SiMe3'], "[O]C(=O)OCC[Si](C)(C)C"),  # 2-(trimethylsilyl)ethoxycarbonyloxy (Teoc carbonate)
    _extra(['OTMSE', 'TMSEO', 'OCH2CH2SiMe3', 'OCH2CH2TMS', 'TMSCH2CH2O', 'Me3SiCH2CH2O', 'O(CH2)2TMS', 'O(CH2)2SiMe3'], "[O]CC[Si](C)(C)C"),  # 2-(trimethylsilyl)ethoxy
    _extra(['OTroc', 'TrocO', 'OCO2CH2CCl3', 'Cl3CCH2OCO2', 'OC(O)OCH2CCl3'], "[O]C(=O)OCC(Cl)(Cl)Cl"),  # 2,2,2-trichloroethoxycarbonyloxy (Troc carbonate)
    _extra(['Pac'], "[CH2]C(=O)c1ccccc1"),  # phenacyl
    _extra(['Px', 'Pixyl'], "[C]1(c2ccccc2)c2ccccc2Oc2ccccc21"),  # 9-phenylxanthen-9-yl (pixyl)
    _extra(['Tce'], "[CH2]C(Cl)(Cl)Cl"),  # 2,2,2-trichloroethyl
    _extra(['TDS'], "[Si](C)(C)C(C)(C)C(C)C"),  # thexyldimethylsilyl
    _extra(['THF', '2-THF'], "[CH]1CCCO1"),  # tetrahydrofuran-2-yl (1 bond)
    _extra(['TIPDS'], "[Si](C(C)C)(C(C)C)O[Si](C(C)C)C(C)C", 2),  # 1,1,3,3-tetraisopropyldisiloxane-1,3-diyl (Markiewicz), in-line between two O atoms
    _extra(['TMSE', 'TMSEt'], "[CH2]CC[Si](C)(C)C"),  # 2-(trimethylsilyl)ethyl
    # s_p_b_si_sn
    _extra(['9-BBN', 'BBN', '9BBN'], "[B]1C2CCCC1CCC2"),  # 9-borabicyclo[3.3.1]nonan-9-yl (9-BBN substituent, 1 bond)
    _extra(['Acm'], "[CH2]NC(C)=O"),  # acetamidomethyl
    _extra(['AlCl2', 'Cl2Al'], "[Al](Cl)Cl"),  # dichloroaluminyl
    _extra(['AlEt2', 'Et2Al'], "[Al](CC)CC"),  # diethylaluminyl
    _extra(['AliBu2', 'iBu2Al', 'Al(iBu)2', 'Al(i-Bu)2', '(i-Bu)2Al'], "[Al](CC(C)C)CC(C)C"),  # diisobutylaluminyl (DIBAL fragment)
    _extra(['AlMe2', 'Me2Al', 'Al(CH3)2'], "[Al](C)C"),  # dimethylaluminyl
    _extra(['AlMeCl', 'ClMeAl'], "[Al](C)Cl"),  # chloro(methyl)aluminyl
    _extra(['B(C6F5)2', '(C6F5)2B'], "[B](c1c(F)c(F)c(F)c(F)c1F)c1c(F)c(F)c(F)c(F)c1F"),  # bis(pentafluorophenyl)boryl
    _extra(['B(C6H4F)2', '(4-FC6H4)2B'], "[B](c1ccc(F)cc1)c1ccc(F)cc1"),  # bis(4-fluorophenyl)boryl
    _extra(['B(Ipc)2', 'Ipc2B', '(Ipc)2B'], "[B](C1CC2C(C)(C)C(C1C)C2)C1CC2C(C)(C)C(C1C)C2"),  # diisopinocampheylboryl (stereo not specified)
    _extra(['B(NiPr2)2', '(iPr2N)2B'], "[B](N(C(C)C)C(C)C)N(C(C)C)C(C)C"),  # bis(diisopropylamino)boryl
    _extra(['B(NMe2)2', '(Me2N)2B'], "[B](N(C)C)N(C)C"),  # bis(dimethylamino)boryl
    _extra(['B(OBu)2', '(BuO)2B', 'B(OnBu)2'], "[B](OCCCC)OCCCC"),  # dibutoxyboryl
    _extra(['B(OEt)2', '(EtO)2B'], "[B](OCC)OCC"),  # diethoxyboryl
    _extra(['B(OH)3-', '-B(OH)3', 'B(OH)3'], "[B-](O)(O)O"),  # trihydroxyborate (boronate anion)
    _extra(['B(OiPr)2', '(iPrO)2B', 'B(Oi-Pr)2', '(i-PrO)2B'], "[B](OC(C)C)OC(C)C"),  # diisopropoxyboryl
    _extra(['B(OMe)2', '(MeO)2B', 'B(OCH3)2'], "[B](OC)OC"),  # dimethoxyboryl
    _extra(['B(Sia)2', 'Sia2B', '(Sia)2B'], "[B](C(C)C(C)C)C(C)C(C)C"),  # disiamylboryl (bis(1,2-dimethylpropyl)boryl)
    _extra(['BBr2', 'Br2B'], "[B](Br)Br"),  # dibromoboryl
    _extra(['BBu2', 'Bu2B', 'BnBu2', 'nBu2B', 'B(nBu)2'], "[B](CCCC)CCCC"),  # dibutylboryl
    _extra(['BC8H14'], "[B]1C2CCCC1CCC2"),  # 9-BBN as a formula (BC8H14)
    _extra(['Bcat', 'B(cat)', 'catB', '(cat)B'], "[B]1Oc2ccccc2O1"),  # catecholboryl
    _extra(['BCl2', 'Cl2B'], "[B](Cl)Cl"),  # dichloroboryl
    _extra(['BCy2', 'Cy2B', 'B(Cy)2'], "[B](C1CCCCC1)C1CCCCC1"),  # dicyclohexylboryl
    _extra(['Bdan', 'B(dan)', 'danB', '(dan)B', 'B-dan'], "[B]1Nc2cccc3cccc(N1)c23"),  # 1,8-diaminonaphthalene boryl (B(dan))
    _extra(['BEt2', 'Et2B', 'B(Et)2'], "[B](CC)CC"),  # diethylboryl
    _extra(['BF2', 'F2B'], "[B](F)F"),  # difluoroboryl (1 bond)
    _extra(['BF3Cs', 'CsF3B'], "[B-](F)(F)F.[Cs+]"),  # caesium trifluoroborate
    _extra(['BF3Li', 'LiF3B'], "[B-](F)(F)F.[Li+]"),  # lithium trifluoroborate
    _extra(['BF3Na', 'NaF3B'], "[B-](F)(F)F.[Na+]"),  # sodium trifluoroborate
    _extra(['BF3NBu4', 'BF3-NBu4+', 'Bu4NF3B'], "[B-](F)(F)F.CCCC[N+](CCCC)(CCCC)CCCC"),  # tetrabutylammonium trifluoroborate
    _extra(['BF3NMe4', 'BF3-NMe4+'], "[B-](F)(F)F.C[N+](C)(C)C"),  # tetramethylammonium trifluoroborate
    _extra(['BHmes2', 'BH(Mes)2', 'Mes2BH-'], "[BH-](c1c(C)cc(C)cc1C)c1c(C)cc(C)cc1C"),  # dimesitylhydridoborate
    _extra(['BMe2', 'Me2B'], "[B](C)C"),  # dimethylboryl
    _extra(['BMes2', 'Bmes2', 'Mes2B', 'B(Mes)2'], "[B](c1c(C)cc(C)cc1C)c1c(C)cc(C)cc1C"),  # dimesitylboryl
    _extra(['BMIDA', 'B(MIDA)', 'MIDAB', '(MIDA)B'], "[B-]12OC(=O)C[N+]1(C)CC(=O)O2"),  # N-methyliminodiacetic acid boronate (MIDA boronate, dative B-N written as B-/N+)
    _extra(['Bneop', 'B(neop)', 'neopB', 'Bneo', 'B(neo)'], "[B]1OCC(C)(C)CO1"),  # neopentyl glycolato boryl (5,5-dimethyl-1,3,2-dioxaborinan-2-yl)
    _extra(['BPh2', 'Ph2B', 'B(Ph)2'], "[B](c1ccccc1)c1ccccc1"),  # diphenylboryl
    _extra(['GeBu3', 'Bu3Ge', 'GenBu3', 'nBu3Ge'], "[Ge](CCCC)(CCCC)CCCC"),  # tributylgermyl
    _extra(['GeEt3', 'Et3Ge'], "[Ge](CC)(CC)CC"),  # triethylgermyl
    _extra(['GeMe3', 'Me3Ge', 'Ge(CH3)3'], "[Ge](C)(C)C"),  # trimethylgermyl
    _extra(['GePh3', 'Ph3Ge'], "[Ge](c1ccccc1)(c1ccccc1)c1ccccc1"),  # triphenylgermyl
    _extra(['H2B'], "[BH2]"),  # boranyl (reversed spelling; BH2 itself is an atom token)
    _extra(['H2P'], "[PH2]"),  # phosphanyl (reversed spelling; PH2 itself is an atom token)
    _extra(['H3B-'], "[BH3-]"),  # borohydride-type BH3(-) attached (amine-borane / phosphine-borane written with a charge)
    _extra(['H3Si'], "[SiH3]"),  # silyl (reversed spelling; SiH3 itself is an atom token)
    _extra(['HSe'], "[SeH]"),  # selanyl (selenol, reversed spelling; SeH itself is an atom token)
    _extra(['Mg(OAc)', 'AcOMg', 'MgOAc'], "[Mg]OC(C)=O"),  # acetoxymagnesio
    _extra(['MgBr', 'BrMg'], "[Mg]Br"),  # bromomagnesio (Grignard fragment)
    _extra(['MgCl', 'ClMg'], "[Mg]Cl"),  # chloromagnesio (Grignard fragment)
    _extra(['MgI', 'IMg'], "[Mg]I"),  # iodomagnesio
    _extra(['MgMe', 'MeMg'], "[Mg]C"),  # methylmagnesio
    _extra(['MgOTf', 'TfOMg'], "[Mg]OS(=O)(=O)C(F)(F)F"),  # (triflyloxy)magnesio
    _extra(['Mob'], "[CH2]c1ccc(OC)cc1"),  # 4-methoxybenzyl (Mob, thiol protecting-group spelling)
    _extra(['P(3,5-(CF3)2C6H3)2'], "[P](c1cc(C(F)(F)F)cc(C(F)(F)F)c1)c1cc(C(F)(F)F)cc(C(F)(F)F)c1"),  # bis(3,5-bis(trifluoromethyl)phenyl)phosphanyl
    _extra(['P(3,5-Xyl)2', '(3,5-Xyl)2P'], "[P](c1cc(C)cc(C)c1)c1cc(C)cc(C)c1"),  # bis(3,5-xylyl)phosphanyl (3,5-dimethylphenyl assumed)
    _extra(['P(4-CF3C6H4)2', 'P(p-CF3C6H4)2'], "[P](c1ccc(C(F)(F)F)cc1)c1ccc(C(F)(F)F)cc1"),  # bis(4-(trifluoromethyl)phenyl)phosphanyl
    _extra(['P(4-MeOC6H4)2', 'P(PMP)2', '(PMP)2P', 'P(p-An)2', 'PAn2'], "[P](c1ccc(OC)cc1)c1ccc(OC)cc1"),  # bis(4-methoxyphenyl)phosphanyl
    _extra(['P(=O)(OEt)2', 'P(=O)(OC2H5)2', '(EtO)2P(=O)', 'P(O)(OC2H5)2', 'PO(OC2H5)2'], "[P](=O)(OCC)OCC"),  # diethoxyphosphoryl (further spellings)
    _extra(['P(=O)(OMe)2', '(MeO)2P(=O)', 'P(O)(OCH3)2', 'PO(OCH3)2'], "[P](=O)(OC)OC"),  # dimethoxyphosphoryl (further spellings)
    _extra(['P(=O)Ph2', 'P(=O)(Ph)2', 'P(O)(Ph)2'], "[P](=O)(c1ccccc1)c1ccccc1"),  # diphenylphosphoryl (further spellings)
    _extra(['P(C6F5)2', '(C6F5)2P'], "[P](c1c(F)c(F)c(F)c(F)c1F)c1c(F)c(F)c(F)c(F)c1F"),  # bis(pentafluorophenyl)phosphanyl
    _extra(['P(NEt2)2', '(Et2N)2P'], "[P](N(CC)CC)N(CC)CC"),  # bis(diethylamino)phosphanyl
    _extra(['P(NiPr2)2', '(iPr2N)2P', 'P(N(iPr)2)2'], "[P](N(C(C)C)C(C)C)N(C(C)C)C(C)C"),  # bis(diisopropylamino)phosphanyl
    _extra(['P(NMe2)2', '(Me2N)2P'], "[P](N(C)C)N(C)C"),  # bis(dimethylamino)phosphanyl
    _extra(['P(NMe2)3+', '(Me2N)3P+', '+P(NMe2)3'], "[P+](N(C)C)(N(C)C)N(C)C"),  # tris(dimethylamino)phosphonio
    _extra(['P(O)(NEt2)2', 'PO(NEt2)2', '(Et2N)2P(O)'], "[P](=O)(N(CC)CC)N(CC)CC"),  # bis(diethylamino)phosphoryl
    _extra(['P(O)(NMe2)2', 'PO(NMe2)2', '(Me2N)2P(O)', '(Me2N)2OP'], "[P](=O)(N(C)C)N(C)C"),  # bis(dimethylamino)phosphoryl
    _extra(['P(O)(o-Tol)2', 'P(O)(o-tol)2', '(o-Tol)2P(O)'], "[P](=O)(c1ccccc1C)c1ccccc1C"),  # di(o-tolyl)phosphoryl
    _extra(['P(O)(OBn)2', 'PO(OBn)2', 'PO3Bn2', '(BnO)2P(O)', '(BnO)2OP', 'P(=O)(OBn)2'], "[P](=O)(OCc1ccccc1)OCc1ccccc1"),  # dibenzyloxyphosphoryl
    _extra(['P(O)(OEt)(OH)', 'P(O)(OH)(OEt)', 'PO(OH)OEt', 'P(O)(OH)OEt', '(EtO)(HO)P(O)', 'PO3HEt'], "[P](=O)(O)OCC"),  # ethoxy(hydroxy)phosphoryl (phosphonate monoester)
    _extra(['P(O)(OEt)H', 'P(O)H(OEt)', 'PH(O)OEt', 'P(O)(H)OEt', 'HP(O)OEt'], "[PH](=O)OCC"),  # ethoxy(hydro)phosphoryl (H-phosphinate)
    _extra(['P(O)(OEt)Me', 'P(O)Me(OEt)', 'MeP(O)OEt', 'P(O)(Me)OEt'], "[P](=O)(OCC)C"),  # ethoxy(methyl)phosphoryl
    _extra(['P(O)(OEt)NHPh', 'P(O)(NHPh)OEt'], "[P](=O)(OCC)Nc1ccccc1"),  # ethoxy(phenylamino)phosphoryl
    _extra(['P(O)(OEt)Ph', 'P(O)Ph(OEt)', 'P(O)(Ph)OEt', 'PhP(O)OEt'], "[P](=O)(OCC)c1ccccc1"),  # ethoxy(phenyl)phosphoryl
    _extra(['P(O)(OH)H', 'PO2H2', 'H2O2P', 'PH(O)OH', 'P(O)H(OH)', 'HP(O)OH'], "[PH](=O)O"),  # hydroxy(hydro)phosphoryl (phosphinic acid)
    _extra(['P(O)(OH)Me', 'P(O)Me(OH)', 'MeP(O)OH', 'P(O)(Me)OH'], "[P](=O)(O)C"),  # hydroxy(methyl)phosphoryl (methylphosphinic acid)
    _extra(['P(O)(OH)Ph', 'P(O)Ph(OH)', 'PhP(O)OH', 'P(O)(Ph)OH'], "[P](=O)(O)c1ccccc1"),  # hydroxy(phenyl)phosphoryl (phenylphosphinic acid)
    _extra(['P(O)(OiPr)2', 'PO(OiPr)2', 'PO3iPr2', '(iPrO)2P(O)', '(iPrO)2OP', 'P(=O)(OiPr)2', 'P(O)(Oi-Pr)2'], "[P](=O)(OC(C)C)OC(C)C"),  # diisopropoxyphosphoryl
    _extra(['P(O)(OMe)(OH)', 'P(O)(OH)(OMe)', 'PO(OH)OMe', 'P(O)(OH)OMe', 'PO3HMe'], "[P](=O)(O)OC"),  # hydroxy(methoxy)phosphoryl
    _extra(['P(O)(OPh)2', 'PO(OPh)2', 'PO3Ph2', '(PhO)2P(O)', '(PhO)2OP', 'P(=O)(OPh)2'], "[P](=O)(Oc1ccccc1)Oc1ccccc1"),  # diphenoxyphosphoryl
    _extra(['P(O)(OPh)NHPh', 'P(O)(NHPh)OPh'], "[P](=O)(Oc1ccccc1)Nc1ccccc1"),  # phenoxy(phenylamino)phosphoryl
    _extra(['P(O)(OtBu)2', 'PO(OtBu)2', 'PO3tBu2', '(tBuO)2P(O)', '(tBuO)2OP'], "[P](=O)(OC(C)(C)C)OC(C)(C)C"),  # di-tert-butoxyphosphoryl
    _extra(['P(O)(p-Tol)2', 'P(O)Tol2', 'Tol2P(O)', 'P(O)(pTol)2'], "[P](=O)(c1ccc(C)cc1)c1ccc(C)cc1"),  # di(p-tolyl)phosphoryl
    _extra(['P(O)Cl2', 'POCl2', 'Cl2P(O)', 'Cl2OP', 'Cl2(O)P', 'P(=O)Cl2'], "[P](=O)(Cl)Cl"),  # dichlorophosphoryl (phosphonic dichloride)
    _extra(['P(O)Cy2', 'P(O)(Cy)2', 'POCy2', 'Cy2P(O)', 'Cy2OP'], "[P](=O)(C1CCCCC1)C1CCCCC1"),  # dicyclohexylphosphoryl
    _extra(['P(O)Et2', 'POEt2', 'Et2P(O)', 'Et2OP'], "[P](=O)(CC)CC"),  # diethylphosphoryl
    _extra(['P(O)iPr2', 'P(O)(iPr)2', 'iPr2P(O)', 'iPr2OP'], "[P](=O)(C(C)C)C(C)C"),  # diisopropylphosphoryl
    _extra(['P(O)Me2', 'POMe2', 'Me2P(O)', 'Me2OP', 'Me2(O)P', 'P(=O)Me2', 'P(O)(CH3)2', '(CH3)2P(O)'], "[P](=O)(C)C"),  # dimethylphosphoryl (dimethylphosphine oxide)
    _extra(['P(O)PhMe', 'P(O)MePh', 'P(O)(Me)Ph', 'MePhP(O)', 'PhMeP(O)', 'P(O)(Ph)Me'], "[P](=O)(C)c1ccccc1"),  # methyl(phenyl)phosphoryl
    _extra(['P(O)tBu2', 'P(O)(tBu)2', 'POtBu2', 'tBu2P(O)', 'tBu2OP', '(tBu)2P(O)'], "[P](=O)(C(C)(C)C)C(C)(C)C"),  # di-tert-butylphosphoryl
    _extra(['P(o-Tol)2', 'P(o-tol)2', '(o-Tol)2P', '(o-tol)2P', 'P(oTol)2', 'P(2-Tol)2'], "[P](c1ccccc1C)c1ccccc1C"),  # di(o-tolyl)phosphanyl
    _extra(['P(o-Tol)3+', '(o-Tol)3P+', 'P(o-tol)3+'], "[P+](c1ccccc1C)(c1ccccc1C)c1ccccc1C"),  # tri(o-tolyl)phosphonio
    _extra(['P(OEt)2', '(EtO)2P'], "[P](OCC)OCC"),  # diethoxyphosphanyl (phosphonite)
    _extra(['P(OEt)3+', '(EtO)3P+'], "[P+](OCC)(OCC)OCC"),  # triethoxyphosphonio (Arbuzov intermediate)
    _extra(['P(OiPr)2', '(iPrO)2P'], "[P](OC(C)C)OC(C)C"),  # diisopropoxyphosphanyl
    _extra(['P(OMe)2', '(MeO)2P'], "[P](OC)OC"),  # dimethoxyphosphanyl (phosphonite)
    _extra(['P(OMe)3+', '(MeO)3P+'], "[P+](OC)(OC)OC"),  # trimethoxyphosphonio
    _extra(['P(OPh)2', '(PhO)2P'], "[P](Oc1ccccc1)Oc1ccccc1"),  # diphenoxyphosphanyl
    _extra(['P(OPh)3+', '(PhO)3P+'], "[P+](Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1"),  # triphenoxyphosphonio
    _extra(['P(p-Tol)2', 'P(pTol)2', '(p-Tol)2P', 'P(4-Tol)2', 'PTol2', 'Tol2P'], "[P](c1ccc(C)cc1)c1ccc(C)cc1"),  # di(p-tolyl)phosphanyl (Tol = p-tolyl)
    _extra(['P(S)(NMe2)2', '(Me2N)2P(S)'], "[P](=S)(N(C)C)N(C)C"),  # bis(dimethylamino)phosphorothioyl
    _extra(['P(S)(OEt)2', 'PS(OEt)2', '(EtO)2P(S)', 'P(=S)(OEt)2'], "[P](=S)(OCC)OCC"),  # diethoxyphosphorothioyl
    _extra(['P(S)(OMe)2', 'PS(OMe)2', '(MeO)2P(S)'], "[P](=S)(OC)OC"),  # dimethoxyphosphorothioyl
    _extra(['P(S)Me2', 'Me2P(S)'], "[P](=S)(C)C"),  # dimethylphosphorothioyl
    _extra(['P(S)Ph2', 'PSPh2', 'Ph2P(S)', 'Ph2(S)P', 'P(=S)Ph2', 'PPh2(S)'], "[P](=S)(c1ccccc1)c1ccccc1"),  # diphenylphosphorothioyl (phosphine sulfide)
    _extra(['P(Se)Ph2', 'PSePh2', 'Ph2P(Se)', 'P(=Se)Ph2'], "[P](=[Se])(c1ccccc1)c1ccccc1"),  # diphenylphosphoroselenoyl (phosphine selenide)
    _extra(['P+Ph3', '+PPh3', 'P(Ph)3+'], "[P+](c1ccccc1)(c1ccccc1)c1ccccc1"),  # triphenylphosphonio (further spellings)
    _extra(['PAd2', 'Ad2P', 'P(Ad)2', 'P(1-Ad)2'], "[P](C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2"),  # di(1-adamantyl)phosphanyl
    _extra(['Pb(OAc)3', '(AcO)3Pb'], "[Pb](OC(C)=O)(OC(C)=O)OC(C)=O"),  # triacetoxyplumbyl (aryllead triacetate)
    _extra(['PbPh3', 'Ph3Pb'], "[Pb](c1ccccc1)(c1ccccc1)c1ccccc1"),  # triphenylplumbyl
    _extra(['PBu2', 'Bu2P', 'PnBu2', 'nBu2P', 'P(nBu)2', 'P(n-Bu)2'], "[P](CCCC)CCCC"),  # dibutylphosphanyl
    _extra(['PBu3+', 'Bu3P+', 'nBu3P+', 'PnBu3+', 'P+Bu3', '+PBu3', 'P(nBu)3+', 'P(n-Bu)3+'], "[P+](CCCC)(CCCC)CCCC"),  # tributylphosphonio
    _extra(['PCl2', 'Cl2P'], "[P](Cl)Cl"),  # dichlorophosphanyl
    _extra(['PCy2BH3', 'PCy2(BH3)', 'Cy2P(BH3)', 'Cy2PBH3', 'P(BH3)Cy2'], "[P+]([BH3-])(C1CCCCC1)C1CCCCC1"),  # borane-protected dicyclohexylphosphanyl
    _extra(['PCy3+', 'Cy3P+', 'P+Cy3', '+PCy3'], "[P+](C1CCCCC1)(C1CCCCC1)C1CCCCC1"),  # tricyclohexylphosphonio
    _extra(['PEt2', 'Et2P', 'P(Et)2'], "[P](CC)CC"),  # diethylphosphanyl
    _extra(['PEt3+', 'Et3P+', 'P+Et3', '+PEt3'], "[P+](CC)(CC)CC"),  # triethylphosphonio
    _extra(['PHCy', 'CyPH', 'PH(Cy)'], "[PH]C1CCCCC1"),  # cyclohexylphosphanyl
    _extra(['PHPh', 'PhPH', 'PH(Ph)', 'P(H)Ph'], "[PH]c1ccccc1"),  # phenylphosphanyl (secondary phosphine)
    _extra(['PHtBu', 'tBuPH', 'PH(tBu)'], "[PH]C(C)(C)C"),  # tert-butylphosphanyl
    _extra(['PinB', '(pin)B', 'B(Pin)'], "[B]1OC(C)(C)C(C)(C)O1"),  # pinacolboryl (further spellings)
    _extra(['PMe2BH3', 'PMe2(BH3)', 'Me2P(BH3)', 'Me2PBH3', 'P(BH3)Me2'], "[P+]([BH3-])(C)C"),  # borane-protected dimethylphosphanyl
    _extra(['PMe3+', 'Me3P+', 'P+Me3', '+PMe3', 'PMe3'], "[P+](C)(C)C"),  # trimethylphosphonio
    _extra(['PMePh', 'PPhMe', 'P(Me)Ph', 'P(Ph)Me', 'MePhP', 'PhMeP'], "[P](C)c1ccccc1"),  # methyl(phenyl)phosphanyl
    _extra(['PMes2', 'Mes2P', 'P(Mes)2'], "[P](c1c(C)cc(C)cc1C)c1c(C)cc(C)cc1C"),  # dimesitylphosphanyl
    _extra(['PO3Et-', 'PO3Et'], "[P](=O)([O-])OCC"),  # ethyl phosphonate monoanion
    _extra(['PO3H-', 'PO3H', 'HO3P-'], "[P](=O)(O)[O-]"),  # hydrogen phosphonate (monoanion)
    _extra(['PO3Na2', 'Na2O3P', 'PO3-Na2'], "[P](=O)([O-])[O-].[Na+].[Na+]"),  # disodium phosphonate
    _extra(['PPh2BH3', 'PPh2(BH3)', 'P(BH3)Ph2', 'H3BPPh2', 'Ph2P(BH3)', 'Ph2PBH3', '(BH3)PPh2'], "[P+]([BH3-])(c1ccccc1)c1ccccc1"),  # borane-protected diphenylphosphanyl (P-BH3 adduct, written as the P+/B- zwitterion)
    _extra(['PPh2Me+', 'PMePh2+', 'Ph2MeP+', 'MePh2P+', '+PPh2Me'], "[P+](C)(c1ccccc1)c1ccccc1"),  # methyldiphenylphosphonio
    _extra(['PPhCy', 'PCyPh', 'P(Ph)Cy', 'P(Cy)Ph'], "[P](C1CCCCC1)c1ccccc1"),  # cyclohexyl(phenyl)phosphanyl
    _extra(['PPhMe2+', 'PMe2Ph+', 'PhMe2P+', 'Me2PhP+', '+PMe2Ph'], "[P+](C)(C)c1ccccc1"),  # dimethylphenylphosphonio
    _extra(['PPhMeBH3', 'PMePh(BH3)', 'P(BH3)MePh', 'MePhP(BH3)'], "[P+]([BH3-])(C)c1ccccc1"),  # borane-protected methyl(phenyl)phosphanyl
    _extra(['PPhtBu', 'PtBuPh', 'P(Ph)tBu', 'P(tBu)Ph', 'tBuPhP'], "[P](C(C)(C)C)c1ccccc1"),  # tert-butyl(phenyl)phosphanyl
    _extra(['PtBu2BH3', 'PtBu2(BH3)', 'tBu2P(BH3)', 'tBu2PBH3', 'P(BH3)tBu2'], "[P+]([BH3-])(C(C)(C)C)C(C)(C)C"),  # borane-protected di-tert-butylphosphanyl
    _extra(['S(NH)Me', 'S(=NH)Me', 'MeS(NH)'], "[S](=N)C"),  # S-methylsulfilimine sulfur
    _extra(['S(O)(NH)Me', 'S(O)(=NH)Me', 'S(=O)(=NH)Me', 'S(O)(NH)CH3', 'MeS(O)(NH)', 'MeS(O)(=NH)', 'S(NH)(O)Me'], "[S](=O)(=N)C"),  # S-methylsulfonimidoyl (sulfoximine)
    _extra(['S(O)(NH)NMe2', 'S(O)(=NH)NMe2'], "[S](=O)(=N)N(C)C"),  # N,N-dimethylsulfonimidamide sulfur
    _extra(['S(O)(NH)Ph', 'S(O)(=NH)Ph', 'S(=O)(=NH)Ph', 'PhS(O)(NH)'], "[S](=O)(=N)c1ccccc1"),  # S-phenylsulfonimidoyl (sulfoximine)
    _extra(['S(O)(NMe)Me', 'S(O)(=NMe)Me', 'S(=O)(=NMe)Me'], "[S](=O)(=NC)C"),  # N,S-dimethylsulfonimidoyl
    _extra(['S(O)2Me', 'S(=O)2Me', 'S(O)2CH3', 'S(=O)(=O)Me', 'S(=O)(=O)CH3', 'MeS(O)2', 'MeS(=O)2', 'Me(O)2S'], "[S](C)(=O)=O"),  # methylsulfonyl (parenthesised spellings)
    _extra(['S(O)C6H4Me', 'S(O)C6H4CH3', 'MeC6H4S(O)'], "[S](=O)c1ccc(C)cc1"),  # p-toluenesulfinyl (formula spellings; the Tol spellings are in gen/n_protect_r1)
    _extra(['S(O)Cl', 'SOCl', 'ClS(O)', 'ClOS'], "[S](=O)Cl"),  # chlorosulfinyl (sulfinyl chloride)
    _extra(['S(O)OMe', 'S(O)OCH3', 'MeOS(O)'], "[S](=O)OC"),  # methoxysulfinyl (methyl sulfinate ester)
    _extra(['SAcm', 'AcmS', 'SCH2NHAc', 'AcNHCH2S'], "[S]CNC(C)=O"),  # acetamidomethylthio (Cys(Acm))
    _extra(['SAllyl', 'SAll', 'AllylS', 'AllS', 'SCH2CH=CH2'], "[S]CC=C"),  # allylthio
    _extra(['SBn', 'BnS', 'SCH2Ph', 'PhCH2S', 'SBzl', 'BzlS', 'SCH2C6H5'], "[S]Cc1ccccc1"),  # benzylthio
    _extra(['SBT', 'BTS'], "[S]c1nc2ccccc2s1"),  # benzothiazol-2-ylthio (Julia-Kocienski BT sulfide)
    _extra(['SBu', 'SnBu', 'nBuS', 'BuS', 'SC4H9'], "[S]CCCC"),  # butylthio
    _extra(['SBz', 'BzS', 'SCOPh', 'SC(O)Ph', 'SC(=O)Ph', 'PhCOS', 'PhC(O)S'], "[S]C(=O)c1ccccc1"),  # benzoylthio
    _extra(['SC(O)NH2', 'SCONH2', 'H2NC(O)S', 'H2NCOS'], "[S]C(N)=O"),  # carbamoylthio
    _extra(['SC(O)NMe2', 'SCONMe2', 'SC(=O)NMe2', 'Me2NC(O)S', 'Me2NCOS', 'Me2NC(=O)S'], "[S]C(=O)N(C)C"),  # dimethylcarbamoylthio (Newman-Kwart S-aryl thiocarbamate)
    _extra(['SC(S)NEt2', 'SC(=S)NEt2', 'Et2NC(S)S', 'Et2NCS2'], "[S]C(=S)N(CC)CC"),  # diethyldithiocarbamate sulfur
    _extra(['SC(S)NMe2', 'SCSNMe2', 'SC(=S)NMe2', 'Me2NC(S)S', 'Me2NCS2', 'Me2NC(=S)S'], "[S]C(=S)N(C)C"),  # dimethyldithiocarbamate sulfur
    _extra(['SC(S)OEt', 'SC(=S)OEt', 'EtOC(S)S', 'EtOCS2'], "[S]C(=S)OCC"),  # O-ethyl xanthate sulfur (RAFT/Chugaev)
    _extra(['SC(S)Ph', 'SC(=S)Ph', 'PhC(S)S', 'PhCS2'], "[S]C(=S)c1ccccc1"),  # dithiobenzoate sulfur (RAFT)
    _extra(['SC(S)SBn', 'BnSC(S)S'], "[S]C(=S)SCc1ccccc1"),  # benzyl trithiocarbonate sulfur
    _extra(['SC(S)SBu', 'SC(=S)SBu', 'BuSC(S)S', 'SC(S)SnBu'], "[S]C(=S)SCCCC"),  # butyl trithiocarbonate sulfur (RAFT)
    _extra(['SC(S)SC12H25', 'SC(=S)SC12H25', 'C12H25SC(S)S'], "[S]C(=S)SCCCCCCCCCCCC"),  # dodecyl trithiocarbonate sulfur (RAFT, DDMAT-type)
    _extra(['SC(S)SMe', 'SC(=S)SMe', 'MeSC(S)S', 'MeSCS2'], "[S]C(=S)SC"),  # methyl trithiocarbonate sulfur
    _extra(['SC2H5', 'C2H5S'], "[S]CC"),  # ethylthio (formula spellings)
    _extra(['SC6F5', 'C6F5S', 'F5C6S'], "[S]c1c(F)c(F)c(F)c(F)c1F"),  # pentafluorophenylthio
    _extra(['SC6H5', 'C6H5S', 'H5C6S'], "[S]c1ccccc1"),  # phenylthio (formula spellings)
    _extra(['SCH2CH2CO2H', 'HO2CCH2CH2S', 'S(CH2)2CO2H'], "[S]CCC(=O)O"),  # 2-carboxyethylthio
    _extra(['SCH2CH2NH2', 'H2NCH2CH2S', 'S(CH2)2NH2'], "[S]CCN"),  # 2-aminoethylthio
    _extra(['SCH2CH2OH', 'HOCH2CH2S', 'S(CH2)2OH'], "[S]CCO"),  # 2-hydroxyethylthio
    _extra(['SCH2CN', 'NCCH2S'], "[S]CC#N"),  # cyanomethylthio
    _extra(['SCH2CO2Et', 'EtO2CCH2S', 'SCH2COOEt'], "[S]CC(=O)OCC"),  # (ethoxycarbonyl)methylthio
    _extra(['SCH2CO2H', 'HO2CCH2S', 'SCH2COOH', 'HOOCCH2S'], "[S]CC(=O)O"),  # carboxymethylthio
    _extra(['SCH2CO2Me', 'MeO2CCH2S', 'SCH2COOMe'], "[S]CC(=O)OC"),  # (methoxycarbonyl)methylthio
    _extra(['SCH3', 'CH3S', 'H3CS'], "[S]C"),  # methylthio (formula spellings)
    _extra(['SCHO', 'OHCS', 'SC(O)H'], "[S]C=O"),  # formylthio
    _extra(['SCN'], "[S]C#N"),  # thiocyanato (attached via S)
    _extra(['SCOEt', 'SC(O)Et', 'EtCOS'], "[S]C(=O)CC"),  # propanoylthio
    _extra(['SCOMe', 'SC(O)Me', 'SC(=O)Me', 'SC(O)CH3', 'SCOCH3', 'SC(=O)CH3', 'MeC(O)S', 'MeC(=O)S', 'MeCOS'], "[S]C(C)=O"),  # acetylthio (formula spellings)
    _extra(['SCy', 'CyS', 'SC6H11'], "[S]C1CCCCC1"),  # cyclohexylthio
    _extra(['Se(O)Ph', 'PhSe(O)'], "[Se](=O)c1ccccc1"),  # phenylseleninyl (selenoxide)
    _extra(['SeAc', 'AcSe', 'SeC(O)Me'], "[Se]C(C)=O"),  # acetylseleno
    _extra(['SeBn', 'BnSe', 'SeCH2Ph'], "[Se]Cc1ccccc1"),  # benzylseleno
    _extra(['SeCF3', 'F3CSe', 'CF3Se'], "[Se]C(F)(F)F"),  # trifluoromethylseleno
    _extra(['SeCN', 'NCSe'], "[Se]C#N"),  # selenocyanato
    _extra(['SeEt', 'EtSe'], "[Se]CC"),  # ethylseleno
    _extra(['SeMe', 'MeSe', 'SeCH3', 'CH3Se', 'H3CSe'], "[Se]C"),  # methylseleno
    _extra(['SeMes', 'MesSe'], "[Se]c1c(C)cc(C)cc1C"),  # mesitylseleno
    _extra(['SeO2H', 'HO2Se', 'Se(O)OH'], "[Se](=O)O"),  # seleninic acid
    _extra(['SePh', 'PhSe', 'SeC6H5', 'C6H5Se'], "[Se]c1ccccc1"),  # phenylseleno
    _extra(['SeSePh', 'PhSeSe'], "[Se][Se]c1ccccc1"),  # phenyldiselanyl
    _extra(['SeTol', 'TolSe', 'SepTol'], "[Se]c1ccc(C)cc1"),  # 4-methylphenylseleno (p assumed)
    _extra(['Si(CH3)3', '(CH3)3Si', '(H3C)3Si', 'Tms'], "[Si](C)(C)C"),  # trimethylsilyl (formula / observed spellings)
    _extra(['Si(OAc)3', '(AcO)3Si'], "[Si](OC(C)=O)(OC(C)=O)OC(C)=O"),  # triacetoxysilyl
    _extra(['Si(OEt)3', '(EtO)3Si', 'Si(OC2H5)3'], "[Si](OCC)(OCC)OCC"),  # triethoxysilyl
    _extra(['Si(OMe)3', '(MeO)3Si', 'Si(OCH3)3'], "[Si](OC)(OC)OC"),  # trimethoxysilyl
    _extra(['Si(SiMe3)3', '(Me3Si)3Si', 'Si(TMS)3', '(TMS)3Si', 'TTMS'], "[Si]([Si](C)(C)C)([Si](C)(C)C)[Si](C)(C)C"),  # tris(trimethylsilyl)silyl (supersilyl)
    _extra(['SiCl3', 'Cl3Si'], "[Si](Cl)(Cl)Cl"),  # trichlorosilyl
    _extra(['SiEt2Cl', 'ClEt2Si'], "[Si](CC)(CC)Cl"),  # chlorodiethylsilyl
    _extra(['SiEt2H', 'HEt2Si', 'SiHEt2'], "[SiH](CC)CC"),  # diethylsilyl (Si-H)
    _extra(['SiEt2iPr', 'iPrEt2Si'], "[Si](CC)(CC)C(C)C"),  # diethylisopropylsilyl (formula spelling)
    _extra(['SiEt2Me', 'Et2MeSi', 'SiMeEt2'], "[Si](C)(CC)CC"),  # diethylmethylsilyl
    _extra(['SiF3', 'F3Si'], "[Si](F)(F)F"),  # trifluorosilyl
    _extra(['SiiPr2H', 'HiPr2Si', 'Si(iPr)2H'], "[SiH](C(C)C)C(C)C"),  # diisopropylsilyl (Si-H)
    _extra(['SiiPr3', 'Si(iPr)3', 'iPr3Si', '(iPr)3Si', 'Si(i-Pr)3', '(i-Pr)3Si', 'SiPri3'], "[Si](C(C)C)(C(C)C)C(C)C"),  # triisopropylsilyl (formula spellings)
    _extra(['SiMe(OMe)2', 'Si(OMe)2Me', '(MeO)2MeSi'], "[Si](C)(OC)OC"),  # dimethoxy(methyl)silyl
    _extra(['SiMe(SiMe3)2', '(Me3Si)2MeSi'], "[Si](C)([Si](C)(C)C)[Si](C)(C)C"),  # bis(trimethylsilyl)methylsilyl
    _extra(['SiMe2Allyl', 'SiMe2allyl', 'SiMe2CH2CH=CH2', 'allylMe2Si'], "[Si](C)(C)CC=C"),  # allyldimethylsilyl
    _extra(['SiMe2Bn', 'BnMe2Si', 'SiMe2CH2Ph'], "[Si](C)(C)Cc1ccccc1"),  # benzyldimethylsilyl
    _extra(['SiMe2CH2Cl', 'ClCH2Me2Si', 'SiMe2(CH2Cl)'], "[Si](C)(C)CCl"),  # (chloromethyl)dimethylsilyl
    _extra(['SiMe2Cl', 'ClMe2Si', 'SiClMe2', 'Me2ClSi'], "[Si](C)(C)Cl"),  # chlorodimethylsilyl
    _extra(['SiMe2Et', 'EtMe2Si'], "[Si](C)(C)CC"),  # ethyldimethylsilyl
    _extra(['SiMe2F', 'FMe2Si', 'SiFMe2'], "[Si](C)(C)F"),  # fluorodimethylsilyl
    _extra(['SiMe2H', 'HMe2Si', 'SiHMe2', 'Me2HSi', 'Me2SiH'], "[SiH](C)C"),  # dimethylsilyl (Si-H)
    _extra(['SiMe2iPr', 'iPrMe2Si', 'SiiPrMe2'], "[Si](C)(C)C(C)C"),  # dimethylisopropylsilyl (formula spelling)
    _extra(['SiMe2OEt', 'EtOMe2Si'], "[Si](C)(C)OCC"),  # ethoxydimethylsilyl
    _extra(['SiMe2OH', 'HOMe2Si', 'SiMe2(OH)', 'Si(OH)Me2'], "[Si](C)(C)O"),  # hydroxydimethylsilyl (silanol)
    _extra(['SiMe2OiPr', 'iPrOMe2Si'], "[Si](C)(C)OC(C)C"),  # isopropoxydimethylsilyl
    _extra(['SiMe2OMe', 'MeOMe2Si', 'Si(OMe)Me2'], "[Si](C)(C)OC"),  # methoxydimethylsilyl
    _extra(['SiMe2OSiMe3', 'Me3SiOMe2Si', 'SiMe2OTMS'], "[Si](C)(C)O[Si](C)(C)C"),  # pentamethyldisiloxanyl
    _extra(['SiMe2OTf', 'TfOMe2Si'], "[Si](C)(C)OS(=O)(=O)C(F)(F)F"),  # dimethyl(triflyloxy)silyl
    _extra(['SiMe2Ph', 'PhMe2Si', 'Me2PhSi', 'SiPhMe2', 'Si(Me)2Ph', 'PhSiMe2'], "[Si](C)(C)c1ccccc1"),  # dimethylphenylsilyl
    _extra(['SiMe2SiMe3', 'Me3SiMe2Si', 'SiMe2TMS', 'TMSMe2Si', 'Si2Me5', 'Me5Si2'], "[Si](C)(C)[Si](C)(C)C"),  # pentamethyldisilanyl
    _extra(['SiMe2tBu', 'SitBuMe2', 'tBuMe2Si', 'SiMe2(tBu)', 'Si(tBu)Me2', 'SiMe2t-Bu', 't-BuMe2Si', 'SiMe2Bu-t', 'Si(t-Bu)Me2', 'SiMe2C(CH3)3'], "[Si](C)(C)C(C)(C)C"),  # tert-butyldimethylsilyl (formula spellings)
    _extra(['SiMe2Vin', 'SiMe2CH=CH2', 'CH2=CHMe2Si', 'SiMe2vinyl'], "[Si](C)(C)C=C"),  # dimethyl(vinyl)silyl
    _extra(['SiMeCl2', 'Cl2MeSi', 'SiCl2Me'], "[Si](C)(Cl)Cl"),  # dichloro(methyl)silyl
    _extra(['SiMeH2', 'H2MeSi', 'SiH2Me'], "[SiH2]C"),  # methylsilyl (SiH2)
    _extra(['SiMePh2', 'Ph2MeSi', 'MePh2Si', 'SiPh2Me', 'Ph2SiMe'], "[Si](C)(c1ccccc1)c1ccccc1"),  # methyldiphenylsilyl
    _extra(['SiPh2Cl', 'ClPh2Si'], "[Si](Cl)(c1ccccc1)c1ccccc1"),  # chlorodiphenylsilyl
    _extra(['SiPh2H', 'HPh2Si', 'SiHPh2'], "[SiH](c1ccccc1)c1ccccc1"),  # diphenylsilyl (Si-H)
    _extra(['SiPh2OH', 'HOPh2Si'], "[Si](O)(c1ccccc1)c1ccccc1"),  # hydroxydiphenylsilyl
    _extra(['SiPr', 'iPrS', 'i-PrS', 'SPr-i', 'SCHMe2', 'SCH(CH3)2'], "[S]C(C)C"),  # isopropylthio
    _extra(['SitBu2Me', 'tBu2MeSi', 'SiMetBu2', 'MetBu2Si', 'Si(tBu)2Me'], "[Si](C)(C(C)(C)C)C(C)(C)C"),  # di-tert-butylmethylsilyl (formula spellings)
    _extra(['SitBuPh2', 'SiPh2tBu', 'tBuPh2Si', 'Si(tBu)Ph2', 'SiPh2(tBu)', 't-BuPh2Si', 'SiPh2t-Bu'], "[Si](c1ccccc1)(c1ccccc1)C(C)(C)C"),  # tert-butyldiphenylsilyl (formula spellings)
    _extra(['SiTol3', 'Tol3Si', 'Si(p-Tol)3'], "[Si](c1ccc(C)cc1)(c1ccc(C)cc1)c1ccc(C)cc1"),  # tri(p-tolyl)silyl
    _extra(['SMe2+', 'S+Me2', 'Me2S+', '+SMe2'], "[S+](C)C"),  # dimethylsulfonio
    _extra(['SMes', 'MesS'], "[S]c1c(C)cc(C)cc1C"),  # mesitylthio
    _extra(['SMob', 'MobS', 'SPMB', 'PMBS'], "[S]Cc1ccc(OC)cc1"),  # 4-methoxybenzylthio
    _extra(['Sn(CH3)3', '(CH3)3Sn', '(H3C)3Sn'], "[Sn](C)(C)C"),  # trimethylstannyl (formula spelling)
    _extra(['Sn(nBu)3', 'Sn(n-Bu)3', '(nBu)3Sn', '(n-Bu)3Sn', 'Sn(Bu)3', '(Bu)3Sn', 'n-Bu3Sn', 'SnBun3', 'Bun3Sn', 'SnC12H27'], "[Sn](CCCC)(CCCC)CCCC"),  # tributylstannyl (further spellings)
    _extra(['SNAC', 'NACS'], "[S]CCNC(C)=O"),  # N-acetylcysteaminyl (SNAC thioester sulfur)
    _extra(['SnBu2Cl', 'ClBu2Sn', 'SnClBu2'], "[Sn](Cl)(CCCC)CCCC"),  # chlorodibutylstannyl
    _extra(['SnBu2OH', 'HOBu2Sn'], "[Sn](O)(CCCC)CCCC"),  # dibutylhydroxystannyl
    _extra(['SnCl3', 'Cl3Sn'], "[Sn](Cl)(Cl)Cl"),  # trichlorostannyl
    _extra(['SnCy3', 'Cy3Sn'], "[Sn](C1CCCCC1)(C1CCCCC1)C1CCCCC1"),  # tricyclohexylstannyl
    _extra(['SnEt3', 'Et3Sn'], "[Sn](CC)(CC)CC"),  # triethylstannyl
    _extra(['SnMe2Cl', 'ClMe2Sn'], "[Sn](Cl)(C)C"),  # chlorodimethylstannyl
    _extra(['SnPh3', 'Ph3Sn', 'Sn(Ph)3'], "[Sn](c1ccccc1)(c1ccccc1)c1ccccc1"),  # triphenylstannyl
    _extra(['SO2Bn', 'BnSO2', 'BnO2S', 'SO2CH2Ph', 'PhCH2SO2', 'PhCH2O2S'], "[S](=O)(=O)Cc1ccccc1"),  # benzylsulfonyl
    _extra(['SO2Br', 'BrSO2', 'BrO2S'], "[S](Br)(=O)=O"),  # bromosulfonyl (sulfonyl bromide)
    _extra(['SO2BT', 'BTSO2'], "[S](=O)(=O)c1nc2ccccc2s1"),  # benzothiazol-2-ylsulfonyl (Julia-Kocienski)
    _extra(['SO2Bu', 'SO2nBu', 'BuSO2', 'nBuSO2'], "[S](=O)(=O)CCCC"),  # butylsulfonyl
    _extra(['SO2C6F5', 'C6F5SO2'], "[S](=O)(=O)c1c(F)c(F)c(F)c(F)c1F"),  # pentafluorophenylsulfonyl
    _extra(['SO2C6H5', 'C6H5SO2', 'H5C6O2S', 'S(O)2Ph', 'S(=O)2Ph', 'S(=O)(=O)Ph', 'PhS(O)2'], "[S](=O)(=O)c1ccccc1"),  # phenylsulfonyl (formula spellings)
    _extra(['SO2CF3', 'CF3SO2', 'F3CSO2', 'F3CO2S', 'CF3O2S', 'S(O)2CF3', 'S(=O)2CF3'], "[S](=O)(=O)C(F)(F)F"),  # trifluoromethylsulfonyl (formula spellings of Tf)
    _extra(['SO2CH2CF3', 'CF3CH2SO2', 'SO2C2H2F3'], "[S](=O)(=O)CC(F)(F)F"),  # 2,2,2-trifluoroethylsulfonyl (formula spelling of Tfes)
    _extra(['SO2CH2CH2SiMe3', 'SO2CH2CH2TMS', 'TMSCH2CH2SO2', 'Me3SiCH2CH2SO2'], "[S](=O)(=O)CC[Si](C)(C)C"),  # 2-(trimethylsilyl)ethylsulfonyl (formula spelling of SES)
    _extra(['SO2CH2SiMe3', 'SO2CH2TMS', 'TMSCH2SO2', 'Me3SiCH2SO2'], "[S](=O)(=O)C[Si](C)(C)C"),  # (trimethylsilyl)methylsulfonyl
    _extra(['SO2Cy', 'CySO2'], "[S](=O)(=O)C1CCCCC1"),  # cyclohexylsulfonyl
    _extra(['SO2Et', 'EtSO2', 'EtO2S', 'SO2C2H5', 'SO2CH2CH3', 'S(O)2Et'], "[S](=O)(=O)CC"),  # ethylsulfonyl
    _extra(['SO2H', 'HO2S', 'S(O)OH', 'SOOH'], "[S](=O)O"),  # sulfino (sulfinic acid)
    _extra(['SO2iPr', 'iPrSO2', 'iPrO2S', 'SO2Pr-i', 'S(O)2iPr'], "[S](=O)(=O)C(C)C"),  # isopropylsulfonyl
    _extra(['SO2Li', 'LiO2S'], "[S](=O)[O-].[Li+]"),  # lithium sulfinate
    _extra(['SO2Mes', 'MesSO2', 'MesO2S'], "[S](=O)(=O)c1c(C)cc(C)cc1C"),  # mesitylsulfonyl
    _extra(['SO2N(CH2CH2)2NMe', 'SO2MePipz', 'MePipzSO2'], "[S](=O)(=O)N1CCN(C)CC1"),  # 4-methylpiperazin-1-ylsulfonyl
    _extra(['SO2N(CH2CH2)2O', 'SO2Morph', 'MorphSO2', 'SO2NC4H8O'], "[S](=O)(=O)N1CCOCC1"),  # morpholin-4-ylsulfonyl
    _extra(['SO2N3', 'N3SO2', 'N3O2S'], "[S](=O)(=O)N=[N+]=[N-]"),  # azidosulfonyl (sulfonyl azide)
    _extra(['SO2Na', 'NaO2S', 'SO2-Na+', 'S(O)ONa'], "[S](=O)[O-].[Na+]"),  # sodium sulfinate
    _extra(['SO2NEt2', 'Et2NSO2', 'Et2NO2S', 'SO2N(Et)2'], "[S](=O)(=O)N(CC)CC"),  # diethylsulfamoyl
    _extra(['SO2NHAc', 'AcHNSO2', 'AcNHSO2', 'SO2NHCOMe', 'SO2NHC(O)Me'], "[S](=O)(=O)NC(C)=O"),  # acetylsulfamoyl (N-acylsulfonamide)
    _extra(['SO2NHBn', 'BnHNSO2', 'BnNHSO2'], "[S](=O)(=O)NCc1ccccc1"),  # benzylsulfamoyl
    _extra(['SO2NHBoc', 'BocHNSO2', 'BocNHSO2'], "[S](=O)(=O)NC(=O)OC(C)(C)C"),  # Boc-sulfamoyl
    _extra(['SO2NHCN', 'NCHNSO2', 'NCNHSO2'], "[S](=O)(=O)NC#N"),  # cyanosulfamoyl
    _extra(['SO2NHCy', 'CyHNSO2', 'CyNHSO2'], "[S](=O)(=O)NC1CCCCC1"),  # cyclohexylsulfamoyl
    _extra(['SO2NHEt', 'EtHNSO2', 'EtNHSO2', 'EtHNO2S'], "[S](=O)(=O)NCC"),  # ethylsulfamoyl
    _extra(['SO2NHiPr', 'iPrHNSO2', 'iPrNHSO2'], "[S](=O)(=O)NC(C)C"),  # isopropylsulfamoyl
    _extra(['SO2NHMe', 'MeHNSO2', 'MeNHSO2', 'MeHNO2S', 'SO2NHCH3', 'CH3NHSO2'], "[S](=O)(=O)NC"),  # methylsulfamoyl (N-methylsulfonamide)
    _extra(['SO2NHNH2', 'H2NHNSO2', 'H2NNHSO2', 'H2NHNO2S'], "[S](=O)(=O)NN"),  # hydrazinylsulfonyl (sulfonyl hydrazide)
    _extra(['SO2NHOH', 'HOHNSO2', 'HONHSO2'], "[S](=O)(=O)NO"),  # hydroxysulfamoyl
    _extra(['SO2NHOMe', 'MeOHNSO2', 'MeONHSO2'], "[S](=O)(=O)NOC"),  # methoxysulfamoyl
    _extra(['SO2NHPh', 'PhHNSO2', 'PhNHSO2', 'PhHNO2S'], "[S](=O)(=O)Nc1ccccc1"),  # phenylsulfamoyl
    _extra(['SO2NHSO2Me', 'MeSO2HNSO2'], "[S](=O)(=O)NS(C)(=O)=O"),  # (methylsulfonyl)sulfamoyl
    _extra(['SO2NHtBu', 'tBuHNSO2', 'tBuNHSO2', 't-BuHNSO2', 'SO2NHBut'], "[S](=O)(=O)NC(C)(C)C"),  # tert-butylsulfamoyl
    _extra(['SO2NMe2', 'Me2NSO2', 'Me2NO2S', 'SO2N(Me)2', 'SO2N(CH3)2', '(CH3)2NSO2', '(H3C)2NO2S'], "[S](=O)(=O)N(C)C"),  # dimethylsulfamoyl
    _extra(['SO2NMe3+', 'Me3N+SO2'], "[S](=O)(=O)[N+](C)(C)C"),  # trimethylammoniosulfonyl (sulfonylammonium)
    _extra(['SO2NMePh', 'SO2N(Me)Ph', 'PhMeNSO2', 'Ph(Me)NSO2'], "[S](=O)(=O)N(C)c1ccccc1"),  # methyl(phenyl)sulfamoyl
    _extra(['SO2OEt', 'SO3Et', 'EtO3S', 'EtOSO2'], "[S](=O)(=O)OCC"),  # ethoxysulfonyl (ethyl sulfonate ester)
    _extra(['SO2OiPr', 'SO3iPr', 'iPrO3S', 'iPrOSO2'], "[S](=O)(=O)OC(C)C"),  # isopropoxysulfonyl
    _extra(['SO2OMe', 'SO3Me', 'MeO3S', 'MeOSO2'], "[S](=O)(=O)OC"),  # methoxysulfonyl (methyl sulfonate ester)
    _extra(['SO2ONp', 'SO3Np', 'SO2OC6H4NO2'], "[S](=O)(=O)Oc1ccc([N+](=O)[O-])cc1"),  # 4-nitrophenoxysulfonyl (4-nitrophenyl sulfonate ester)
    _extra(['SO2OPh', 'SO3Ph', 'PhO3S', 'PhOSO2'], "[S](=O)(=O)Oc1ccccc1"),  # phenoxysulfonyl (phenyl sulfonate ester)
    _extra(['SO2Pip', 'PipSO2', 'SO2NC5H10'], "[S](=O)(=O)N1CCCCC1"),  # piperidin-1-ylsulfonyl
    _extra(['SO2PMP', 'PMPSO2', 'SO2C6H4OMe', 'MeOC6H4SO2'], "[S](=O)(=O)c1ccc(OC)cc1"),  # 4-methoxyphenylsulfonyl (Mbs; p assumed)
    _extra(['SO2Pr', 'SO2nPr', 'PrSO2', 'nPrSO2'], "[S](=O)(=O)CCC"),  # propylsulfonyl
    _extra(['SO2PT', 'PTSO2'], "[S](=O)(=O)c1nnnn1-c1ccccc1"),  # 1-phenyl-1H-tetrazol-5-ylsulfonyl (Julia-Kocienski)
    _extra(['SO2Py', '2-PySO2', 'PySO2'], "[S](=O)(=O)c1ccccn1"),  # pyridin-2-ylsulfonyl (Julia-Kocienski)
    _extra(['SO2Pyrr', 'SO2Pyrrol', 'SO2NC4H8'], "[S](=O)(=O)N1CCCC1"),  # pyrrolidin-1-ylsulfonyl
    _extra(['SO2TBT', 'TBTSO2'], "[S](=O)(=O)c1nnnn1C(C)(C)C"),  # 1-tert-butyl-1H-tetrazol-5-ylsulfonyl (Julia-Kocienski)
    _extra(['SO2tBu', 'tBuSO2', 't-BuSO2', 'tBuO2S', 'SO2t-Bu', 'SO2But', 'S(O)2tBu'], "[S](=O)(=O)C(C)(C)C"),  # tert-butylsulfonyl (Bus)
    _extra(['SO2Tol', 'TolSO2', 'TolO2S', 'SO2pTol', 'pTolSO2', 'SO2p-Tol', 'p-TolSO2', 'SO2C6H4Me', 'SO2C6H4CH3', 'MeC6H4SO2', 'S(O)2Tol'], "[S](=O)(=O)c1ccc(C)cc1"),  # p-toluenesulfonyl (formula spellings of Ts)
    _extra(['SO3Ag', 'AgO3S'], "[S](=O)(=O)[O-].[Ag+]"),  # silver sulfonate
    _extra(['SO3Cs', 'CsO3S'], "[S](=O)(=O)[O-].[Cs+]"),  # caesium sulfonate
    _extra(['SO3Li', 'LiO3S'], "[S](=O)(=O)[O-].[Li+]"),  # lithium sulfonate
    _extra(['SO3NBu4', 'SO3-NBu4+', 'Bu4NO3S'], "[S](=O)(=O)[O-].CCCC[N+](CCCC)(CCCC)CCCC"),  # tetrabutylammonium sulfonate
    _extra(['SO3NEt3H', 'SO3HNEt3', 'Et3NHO3S'], "[S](=O)(=O)[O-].CC[NH+](CC)CC"),  # triethylammonium sulfonate
    _extra(['SOBn', 'S(O)Bn', 'BnS(O)'], "[S](=O)Cc1ccccc1"),  # benzylsulfinyl
    _extra(['SOCF3', 'S(O)CF3', 'CF3S(O)', 'F3CS(O)', 'CF3SO'], "[S](=O)C(F)(F)F"),  # trifluoromethylsulfinyl
    _extra(['SOEt', 'S(O)Et', 'EtS(O)', 'EtSO'], "[S](=O)CC"),  # ethylsulfinyl
    _extra(['SOiPr', 'S(O)iPr', 'iPrS(O)'], "[S](=O)C(C)C"),  # isopropylsulfinyl
    _extra(['SOMe', 'S(O)Me', 'S(=O)Me', 'MeS(O)', 'MeSO', 'SOCH3', 'S(O)CH3', 'CH3S(O)', 'MeOS', 'MeS(=O)'], "[S](=O)C"),  # methylsulfinyl
    _extra(['SOPh', 'S(O)Ph', 'S(=O)Ph', 'PhS(O)', 'PhSO', 'PhOS'], "[S](=O)c1ccccc1"),  # phenylsulfinyl
    _extra(['SPh2+', 'Ph2S+'], "[S+](c1ccccc1)c1ccccc1"),  # diphenylsulfonio
    _extra(['SPiv', 'PivS', 'SCOtBu', 'SC(O)tBu', 'tBuCOS', 'tBuC(O)S'], "[S]C(=O)C(C)(C)C"),  # pivaloylthio
    _extra(['SPMP', 'PMPS', 'SC6H4OMe', 'MeOC6H4S'], "[S]c1ccc(OC)cc1"),  # 4-methoxyphenylthio (p assumed)
    _extra(['SPr', 'SnPr', 'nPrS', 'PrS', 'SC3H7', 'SCH2CH2CH3'], "[S]CCC"),  # propylthio
    _extra(['SPT', 'PTS'], "[S]c1nnnn1-c1ccccc1"),  # 1-phenyl-1H-tetrazol-5-ylthio (Julia-Kocienski PT sulfide)
    _extra(['SPy', 'PyS', '2-PyS', 'SPy-2'], "[S]c1ccccn1"),  # pyridin-2-ylthio (2-pyridylthio; thioesters, disulfides)
    _extra(['SPym', 'PymS'], "[S]c1ncccn1"),  # pyrimidin-2-ylthio
    _extra(['SSBn', 'BnSS'], "[S]SCc1ccccc1"),  # benzyldisulfanyl
    _extra(['SSEt', 'EtSS'], "[S]SCC"),  # ethyldisulfanyl
    _extra(['SSMe', 'MeSS', 'SSCH3', 'CH3SS'], "[S]SC"),  # methyldisulfanyl
    _extra(['SSO3-', '-O3SS'], "[S]S(=O)(=O)[O-]"),  # thiosulfate (Bunte salt) sulfur
    _extra(['SSPh', 'PhSS'], "[S]Sc1ccccc1"),  # phenyldisulfanyl
    _extra(['SStBu', 'tBuSS', 't-BuSS', 'SSBut'], "[S]SC(C)(C)C"),  # tert-butyldisulfanyl (Cys(StBu))
    _extra(['STBS', 'TBSS', 'SSiMe2tBu', 'tBuMe2SiS'], "[S][Si](C)(C)C(C)(C)C"),  # tert-butyldimethylsilylthio
    _extra(['StBu', 'SBut', 'SBu-t', 'tBuS', 't-BuS', 'SCMe3', 'SC(CH3)3'], "[S]C(C)(C)C"),  # tert-butylthio
    _extra(['STMS', 'TMSS', 'SSiMe3', 'Me3SiS'], "[S][Si](C)(C)C"),  # trimethylsilylthio
    _extra(['STol', 'TolS', 'SpTol', 'p-TolS', 'STol-p', 'SC6H4Me', 'SC6H4CH3', 'MeC6H4S'], "[S]c1ccc(C)cc1"),  # 4-methylphenylthio (p-tolylthio; p assumed)
    _extra(['STrt', 'TrtS', 'STr', 'TrS', 'SCPh3', 'Ph3CS'], "[S]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # tritylthio
    _extra(['TeBu', 'TenBu', 'BuTe', 'nBuTe'], "[Te]CCCC"),  # butyltelluro
    _extra(['TeMe', 'MeTe'], "[Te]C"),  # methyltelluro
    _extra(['TePh', 'PhTe', 'TeC6H5'], "[Te]c1ccccc1"),  # phenyltelluro
    _extra(['TeTol', 'TolTe'], "[Te]c1ccc(C)cc1"),  # 4-methylphenyltelluro
    # --- END GENERATED ABBREVIATIONS ---
]

# label -> default entry (n_attach == 1 when available); (label, n_attach) -> entry for bond-count-specific lookup
ABBREVIATIONS_BY_ATTACH = {(abbrv, sub.n_attach): sub for sub in SUBSTITUTIONS + EXTRA_SUBSTITUTIONS
                           for abbrv in sub.abbrvs}
ABBREVIATIONS = {}
for _sub in SUBSTITUTIONS + EXTRA_SUBSTITUTIONS:
    for _abbrv in _sub.abbrvs:
        if _abbrv not in ABBREVIATIONS or (ABBREVIATIONS[_abbrv].n_attach != 1 and _sub.n_attach == 1):
            ABBREVIATIONS[_abbrv] = _sub

# Case-insensitive fallback for labels drawn in all caps (BOC, CBZ, FMOC, NHBOC ...). Only keys of >= 3 chars are
# considered, and only all-uppercase labels are looked up, so element symbols such as Co/Cs/Sn are never affected.
UPPERCASE_ABBREVIATIONS = {}
for _abbrv in ABBREVIATIONS:
    if len(_abbrv) >= 3 and _abbrv.upper() != _abbrv:
        UPPERCASE_ABBREVIATIONS.setdefault(_abbrv.upper(), _abbrv)

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U",
}

# Neutral/anionic ligands used when the label is bonded only to metal atoms; atom 0 is the donor atom and is
# connected to the metal with a dative bond.
LIGAND_SMILES = {
    'CO': "[C-]#[O+]", 'OC': "[C-]#[O+]",
    'CN': "[C-]#N", 'NC': "[C-]#N",
    'PPh3': "P(c1ccccc1)(c1ccccc1)c1ccccc1", 'Ph3P': "P(c1ccccc1)(c1ccccc1)c1ccccc1",
    'PCy3': "P(C1CCCCC1)(C1CCCCC1)C1CCCCC1", 'Cy3P': "P(C1CCCCC1)(C1CCCCC1)C1CCCCC1",
    'PMe3': "P(C)(C)C", 'Me3P': "P(C)(C)C",
    'MeCN': "N#CC", 'NCMe': "N#CC", 'CH3CN': "N#CC", 'NCCH3': "N#CC",
    'tBuNC': "[C-]#[N+]C(C)(C)C", 'CNtBu': "[C-]#[N+]C(C)(C)C",
    'H2O': "O", 'OH2': "O",
    'NH3': "N", 'H3N': "N",
    'THF': "O1CCCC1",
    'py': "n1ccccc1",
    'DMSO': "O=S(C)C",
}

VALENCES = {
    "H": [1], "D": [1], "T": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5, 3], "S": [6, 2, 4], "Cl": [1], "K": [1], "Ca": [2],
    "Br": [1], "I": [1],
    # metals / metalloids that appear in condensed labels (MgBr, ZnCl, SnMe3, HgCl, SePh, B(OH)2 ...)
    "Zn": [2], "Cu": [1, 2], "Sn": [4, 2], "Hg": [2, 1], "Pd": [2], "Pt": [2, 4], "Ni": [2], "Fe": [2, 3],
    "Co": [2, 3], "Ag": [1], "Au": [1, 3], "Cs": [1], "Rb": [1], "Ba": [2], "Sr": [2], "Ti": [4], "Zr": [4],
    "Mn": [2], "Cr": [3], "Cd": [2], "Ga": [3], "In": [3], "Tl": [1, 3], "Pb": [4, 2], "Bi": [3],
    "Ge": [4], "As": [3, 5], "Sb": [3, 5], "Se": [2, 4, 6], "Te": [2, 4, 6],
}

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}

def _longest_first(tokens):
    return '|'.join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))


# Elements that do not occur in drawn structures; a label reading like one of them (e.g. "ZrNp" for "2-Np") is a
# misread and must not be parsed into a formula.
IMPLAUSIBLE_ELEMENTS = {
    "He", "Ne", "Kr", "Xe", "Rn", "Po", "At", "Fr", "Tc", "Pm", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Og",
}

# tokens of condensed formula. Alternatives are tried in order, so longer tokens must come first (otherwise e.g.
# "SO2NH2" is split as S + O2N + H2). Real element symbols are used instead of [A-Z][a-z]+ so that "OiPr" is not
# tokenized as the fake element "Oi". Callers should reject formulas whose tokens do not cover the whole string.
# Only single-bond entries take part in formula tokenization: inside a formula an abbreviation token is treated
# as a monovalent unit, so in-line (2-bond) keys such as C(O) or SO2 must keep being parsed atom by atom.
# A dictionary token or an element symbol may not end right before a lowercase letter: that would split an element
# symbol ("CH2Br" must not become H2B + r).
FORMULA_REGEX = re.compile(
    '((?:' + _longest_first([k for k, sub in ABBREVIATIONS.items() if sub.n_attach == 1]) + ')(?![a-z])|' +
    _longest_first([r for r in RGROUP_SYMBOLS if len(r) > 1]) +
    r"|R[0-9]*['′]*[α-ω]?|(?:" + _longest_first([e for e in ELEMENTS if e not in IMPLAUSIBLE_ELEMENTS]) +
    r')(?![a-z])|D|T|[A-Z]|[0-9]+|\(|\))')
