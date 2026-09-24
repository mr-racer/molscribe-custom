from typing import List
import re

ORGANIC_SET = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']

PLACEHOLDER_ATOMS = ["Lv", "Lu", "Nd", "Yb", "At", "Fm", "Er"]


class Substitution(object):
    '''Define common substitutions for chemical shorthand'''
    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability


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


def _extra(abbrvs, smiles):
    # Inference-only: expanded when predicted, never rendered into synthetic training images (no SMARTS).
    return Substitution(abbrvs, None, smiles, 0.0)


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
    _extra(['Tol', 'p-Tol', 'pTol', '4-Tol'], "[c]1ccc(C)cc1"),
    _extra(['o-Tol', 'oTol', '2-Tol'], "[c]1ccccc1C"),
    _extra(['m-Tol', 'mTol', '3-Tol'], "[c]1cccc(C)c1"),
    _extra(['Dipp'], "[c]1c(C(C)C)cccc1C(C)C"),
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
    _extra(['PPh2', 'Ph2P'], "[P](c1ccccc1)c1ccccc1"),
    _extra(['P(O)Ph2', 'Ph2P(O)', 'POPh2', 'Ph2OP'], "[P](=O)(c1ccccc1)c1ccccc1"),
    _extra(['P(O)(OEt)2', '(EtO)2P(O)', 'PO(OEt)2', '(EtO)2OP', 'PO3Et2'], "[P](=O)(OCC)OCC"),
    _extra(['P(O)(OMe)2', '(MeO)2P(O)', 'PO(OMe)2', '(MeO)2OP', 'PO3Me2'], "[P](=O)(OC)OC"),
    _extra(['PPh3+', 'Ph3P+'], "[P+](c1ccccc1)(c1ccccc1)c1ccccc1"),

    # Counter-ions (drawn without bonds)
    _extra(['BF4', 'BF4-'], "F[B-](F)(F)F"),
    _extra(['PF6', 'PF6-'], "F[P-](F)(F)(F)(F)F"),
    _extra(['SbF6', 'SbF6-'], "F[Sb-](F)(F)(F)(F)F"),
    _extra(['AsF6', 'AsF6-'], "F[As-](F)(F)(F)(F)F"),
    _extra(['ClO4', 'ClO4-'], "[O-][Cl+3]([O-])([O-])[O-]"),
    _extra(['OTf-', 'TfO-', 'CF3SO3-', 'CF3SO3'], "[O-]S(=O)(=O)C(F)(F)F"),
    _extra(['NTf2-', 'Tf2N-'], "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"),
    _extra(['OTs-', 'TsO-'], "[O-]S(=O)(=O)c1ccc(C)cc1"),
]

ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS + EXTRA_SUBSTITUTIONS for abbrv in sub.abbrvs}

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
    "H": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
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


# tokens of condensed formula. Alternatives are tried in order, so longer tokens must come first (otherwise e.g.
# "SO2NH2" is split as S + O2N + H2). Real element symbols are used instead of [A-Z][a-z]+ so that "OiPr" is not
# tokenized as the fake element "Oi". Callers should reject formulas whose tokens do not cover the whole string.
FORMULA_REGEX = re.compile(
    '(' + _longest_first(ABBREVIATIONS) + '|' + _longest_first([r for r in RGROUP_SYMBOLS if len(r) > 1]) +
    r'|R[0-9]*|' + _longest_first(ELEMENTS) + r'|[A-Z]|[0-9]+|\(|\))')
