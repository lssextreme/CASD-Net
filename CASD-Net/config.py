# -*- coding: utf-8 -*-
"""
Dataset registry for CASD-Net.

Each dataset is described by a :class:`DatasetConfig` that bundles its class
taxonomy (``labels`` / ``palette``), its file templates for imagery, DSM and
labels, the official train/test splits, and the training hyper-parameters used
in the CASD-Net paper.

Only ``root`` needs to be edited per machine: point it at the folder that
contains the dataset (Vaihingen / Potsdam / Hunan). Nothing else in the code
depends on an absolute path.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Color palettes
# -----------------------------------------------------------------------------
# ISPRS Vaihingen & Potsdam 6-class palette (the 7th entry, index 6, is the
# "undefined/black" colour present in the raw labels; it never appears in the
# 6-class segmentation output).
ISPRS_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),   # impervious surfaces
    1: (0, 0, 255),       # building
    2: (0, 255, 255),     # low vegetation
    3: (0, 255, 0),       # tree
    4: (255, 255, 0),     # car
    5: (255, 0, 0),       # clutter / background
    6: (0, 0, 0),         # undefined (out-of-range label)
}

# Hunan 7-class palette (as used in the pre-processing pipeline).
HUNAN_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (196, 90, 17),     # cropland
    1: (51, 129, 88),     # forest
    2: (177, 205, 61),    # grassland
    3: (228, 84, 96),     # wetland
    4: (91, 154, 214),    # water
    5: (225, 174, 110),   # unused land
    6: (239, 159, 2),     # built-up area
}


@dataclass
class DatasetConfig:
    """A self-contained description of one training/evaluation dataset."""

    name: str
    root: str
    labels: List[str]
    palette: Dict[int, Tuple[int, int, int]]

    train_ids: List[str]
    test_ids: List[str]

    # File templates relative to ``root``; ``{id}`` is the tile identifier.
    data_template: str
    dsm_template: str
    label_template: str
    eroded_template: str

    # DSM pre-processing at load time:
    #   'local'  -> per-tile min-max normalisation to [0, 1]  (ISPRS)
    #   'none'   -> already globally normalised                 (Hunan)
    dsm_normalization: str = 'local'

    # Label encoding on disk:
    #   'color' -> RGB palette image, decoded through ``palette``  (ISPRS)
    #   'index' -> single-channel integer labels                   (Hunan)
    label_format: str = 'color'

    # Training hyper-parameters (paper defaults; tunable).
    window_size: Tuple[int, int] = (256, 256)
    batch_size: int = 16
    stride: int = 32                 # sliding-window stride at inference
    epochs: int = 150
    save_epoch: int = 10
    steps_per_epoch: int = 300       # fixed patch-sampling budget per epoch


# -----------------------------------------------------------------------------
# Instantiated registries.
# -----------------------------------------------------------------------------
# NOTE: set ``root`` to the local folder containing each dataset. The templates
# below follow the canonical ISPRS layout and the Hunan pre-processing layout
# described in the paper.
VAIHINGEN = DatasetConfig(
    name='Vaihingen',
    root='',                       # e.g. r'D:/data/isprs'
    labels=['Imp. surfaces', 'Building', 'Low veg.', 'Tree', 'Car', 'Clutter'],
    palette=ISPRS_PALETTE,
    train_ids=['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34',
               '37', '5', '21', '15', '30'],
    test_ids=['5', '21', '15', '30'],
    data_template='Vaihingen/top/top_mosaic_09cm_area{id}.tif',
    dsm_template='Vaihingen/dsm/dsm_09cm_matching_area{id}.tif',
    label_template='Vaihingen/gts_for_participants/top_mosaic_09cm_area{id}.tif',
    eroded_template='Vaihingen/gts_eroded_for_participants/top_mosaic_09cm_area{id}_noBoundary.tif',
    dsm_normalization='local',
    label_format='color',
)

POTSDAM = DatasetConfig(
    name='Potsdam',
    root='',                       # e.g. r'D:/data/potsdam'
    labels=['Imp. surfaces', 'Building', 'Low veg.', 'Tree', 'Car', 'Clutter'],
    palette=ISPRS_PALETTE,
    train_ids=['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12',
               '5_12', '7_11', '7_9', '6_9', '7_7', '4_12', '6_8', '6_12',
               '6_7', '4_11', '4_10', '5_11', '2_11', '3_10', '6_11', '7_12'],
    test_ids=['4_10', '5_11', '2_11', '3_10', '6_11', '7_12'],
    data_template='2_Ortho_RGB/2_Ortho_RGB/top_potsdam_{id}_RGB.tif',
    dsm_template='1_DSM_normalisation/1_DSM_normalisation/dsm_potsdam_{id}_normalized_lastools.jpg',
    label_template='5_Labels_for_participants/5_Labels_for_participants/top_potsdam_{id}_label.tif',
    eroded_template='5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/top_potsdam_{id}_label_noBoundary.tif',
    dsm_normalization='local',
    label_format='color',
)

HUNAN = DatasetConfig(
    name='Hunan',
    root='',                       # e.g. r'D:/data/HN/Hunan_Dataset_cor_process1'
    labels=['cropland', 'forest', 'grassland', 'wetland', 'water',
            'unused land', 'built-up area'],
    palette=HUNAN_PALETTE,
    # The official split used in the paper (train/test).
    train_ids=['10434', '11524', '11607', '11724', '11854', '11856', '11919', '12152', '12350', '12563',
               '12669', '12813', '1302', '13258', '13383', '13524', '13565', '13932', '14477', '14694',
               '15001', '15023', '15201', '15230', '15548', '15603', '15686', '1599', '15998', '16090',
               '16217', '16541', '16582', '16703', '16709', '17092', '17269', '18186', '18950', '1899',
               '1906', '19098', '19680', '19915', '20175', '20386', '20561', '20734', '20752', '20759',
               '21562', '21565', '21738', '21820', '22000', '2232', '22547', '22729', '2431', '24718',
               '24733', '25069', '2584', '2617', '26622', '27163', '27308', '27312', '2791', '29393',
               '29886', '30560', '30719', '31175', '31188', '31586', '31797', '3817', '4529', '4530',
               '4889', '5223', '6213', '6597', '6600', '6768', '7329', '8232', '830', '8931',
               '944', '9956', '2087', '17721', '13990', '13622', '13563', '18009', '12148', '16888',
               '14758', '1773', '16516', '20408', '2070', '10062', '17637', '14942', '13931', '13410',
               '11959', '15150', '17582', '17820', '21545', '21563', '21592', '21922', '2255', '26628',
               '28533', '28801', '29621', '29796', '30482', '31302', '31355', '4171', '4887', '5994',
               '6167', '6777', '7421', '833', '9064', '9662', '14936', '15493', '2097', '25709',
               '30301', '18117', '11766', '3994', '5830', '14786', '1774', '16032', '2597', '18164',
               '8976', '2427', '418', '23961', '1165', '6383', '22906', '26032', '18371', '6156',
               '7167', '20736', '16880', '29145', '21211', '7473', '29172', '22077', '14755', '2428',
               '16922', '15144', '5232', '25777', '21736', '14290', '15275', '1025', '11173', '12040',
               '12779', '14126', '15695', '16214', '16577', '18079', '1930', '21804', '22154', '25699',
               '29675', '298', '31653', '5042', '637', '6581', '708', '7679', '1031', '11272',
               '14463', '16745', '12244', '1775', '1752', '16744', '17095', '20910', '13742', '16702',
               '13925', '17800', '17040', '2062', '16912', '19149', '11371', '21601', '21610', '21737',
               '21747', '21982', '23232', '2606', '27860', '28532', '28933', '29028', '29284', '29288',
               '30597', '3122', '31242', '4362', '6405', '6770', '726', '7330', '7331', '8234',
               '2402', '646', '13718', '12363', '13995', '13807', '1471', '27168', '18298', '16093',
               '15763', '12042', '29020', '8831', '11375', '23772', '12728', '13448', '27960', '14467',
               '14763', '19866', '13766', '24296', '1436', '5236', '28796', '10258', '28736', '2100',
               '1451', '12918', '14155', '15184', '19471', '21822', '22728', '22837', '2408', '3100',
               '5450', '6186', '7181', '9258', '11625', '11644', '12098', '12154', '12241', '13248',
               '13522', '13564', '1383', '13927', '14287', '14822', '15075', '15520', '15864', '16028',
               '1619', '16699', '16742', '16866', '17369', '17934', '18183', '18185', '18732', '1928',
               '19688', '20314', '20464', '20737', '21111', '21561', '21938', '22555', '23080', '23588',
               '23701', '2604', '27282', '27718', '28073', '28189', '29171', '29286', '29844', '30148',
               '30399', '30425', '30606', '31014', '31583', '31621', '4363', '5043', '5221', '5238',
               '6379', '6407', '6601', '9174', '2748', '29829', '10970', '20926', '6795', '24149',
               '18121', '20935', '942', '29391', '29638', '20054', '3161', '6772', '17933', '13535',
               '5412', '20599', '299', '19609', '452', '28191', '11659', '1450', '13019', '11838',
               '29892', '12151', '13933', '11568', '11233', '12153', '13433', '13436', '14105', '14169',
               '16724', '18895', '19853', '21981', '2246', '22907', '29654', '30669', '3534', '6794',
               '1057', '11271', '11603', '11678', '11957', '1262', '12863', '13109', '13299', '13415',
               '14603', '14873', '15398', '15596', '1605', '16089', '16530', '16870', '17799', '17943',
               '1932', '21740', '21967', '23342', '2396', '2397', '2429', '2542', '25778', '2618',
               '26776', '28038', '29104', '29567', '29733', '29779', '30276', '31708', '3936', '419',
               '5066', '7898', '20933', '15597', '18118', '16356', '14937', '10503', '13437', '14247',
               '21566', '22099', '22731', '27437', '28078', '29394', '29571', '30130', '6771', '940',
               '11767', '11816', '1239', '12626', '12815', '1290', '1303', '13254', '13257', '13515',
               '13765', '14108', '14293', '15426', '1625', '16373', '16750', '16890', '17039', '17055',
               '17107', '17455', '17821', '17980', '18958', '1908', '1923', '20028', '2106', '21385',
               '21423', '22312', '2256', '2265', '2442', '2603', '27264', '28965', '29287', '30275',
               '3280', '4017', '4166', '4886', '5215', '5233', '5410', '639', '7176', '11624',],
    test_ids=['11767', '11816', '1239', '12626', '12815', '1290', '1303', '13254', '13257', '13515',
              '13765', '14108', '14293', '15426', '1625', '16373', '16750', '16890', '17039', '17055',
              '17107', '17455', '17821', '17980', '18958', '1908', '1923', '20028', '2106', '21385',
              '21423', '22312', '2256', '2265', '2442', '2603', '27264', '28965', '29287', '30275',
              '3280', '4017', '4166', '4886', '5215', '5233', '5410', '639', '7176', '11624',],
    data_template='images_png/{id}.tif',
    dsm_template='dsm_pngs/{id}.tif',
    label_template='masks_png/{id}.tif',
    eroded_template='masks_png/{id}.tif',
    dsm_normalization='none',
    label_format='index',
    batch_size=1,                 # Hunan uses whole pre-tiled images
    steps_per_epoch=400,
)

DATASETS: Dict[str, DatasetConfig] = {
    'Vaihingen': VAIHINGEN,
    'Potsdam': POTSDAM,
    'Hunan': HUNAN,
}
