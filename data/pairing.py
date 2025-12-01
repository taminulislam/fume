"""
Data Pairing Module for FUME
Creates CO2-CH4 paired samples from the dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GasPairCreator:
    """
    Creates paired CO2-CH4 samples from the dataset.

    Strategy:
    - pH 5.6 (Acidotic): 798 CH4 + 1,617 CO2 samples
    - pH 6.5 (Healthy): 1,995 CH4 + 1,911 CO2 samples
    - For unpaired pH levels (5.0, 5.3, 5.9, 6.2): use zero-padding
    """

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.train_csv = self.dataset_root / "train_annotations.csv"
        self.val_csv = self.dataset_root / "val_annotations.csv"
        self.test_csv = self.dataset_root / "test_annotations.csv"

    def load_annotations(self, split: str = 'train') -> pd.DataFrame:
        """Load annotations for a given split"""
        csv_map = {
            'train': self.train_csv,
            'val': self.val_csv,
            'test': self.test_csv
        }
        return pd.read_csv(csv_map[split])

    def create_pairs(self, df: pd.DataFrame, pairing_strategy: str = 'random') -> List[Dict]:
        """
        Create CO2-CH4 pairs from dataframe

        Args:
            df: Annotations dataframe
            pairing_strategy: 'random', 'original_based', or 'augmentation_matched'

        Returns:
            List of paired samples with format:
            {
                'co2_frame': path,
                'co2_mask': path,
                'ch4_frame': path,
                'ch4_mask': path,
                'ph_value': float,
                'class_id': int,
                'class_name': str,
                'is_paired': bool,  # True if both gases available
                'missing_modality': None or 'co2' or 'ch4'
            }
        """
        pairs = []

        # Group by pH value and class
        grouped = df.groupby(['ph_value', 'class_name'])

        for (ph, class_name), group in grouped:
            co2_samples = group[group['gas_type'] == 'co2']
            ch4_samples = group[group['gas_type'] == 'ch4']

            if len(co2_samples) > 0 and len(ch4_samples) > 0:
                # Paired case (pH 5.6 and 6.5)
                pairs.extend(self._create_paired_samples(
                    co2_samples, ch4_samples, ph, class_name, pairing_strategy
                ))
            elif len(co2_samples) > 0:
                # Only CO2 available
                pairs.extend(self._create_unpaired_samples(
                    co2_samples, ph, class_name, missing='ch4'
                ))
            elif len(ch4_samples) > 0:
                # Only CH4 available
                pairs.extend(self._create_unpaired_samples(
                    ch4_samples, ph, class_name, missing='co2'
                ))

        logger.info(f"Created {len(pairs)} total pairs")
        paired_count = sum(1 for p in pairs if p['is_paired'])
        logger.info(f"  - {paired_count} fully paired samples")
        logger.info(f"  - {len(pairs) - paired_count} unpaired samples")

        return pairs

    def _create_paired_samples(
        self,
        co2_samples: pd.DataFrame,
        ch4_samples: pd.DataFrame,
        ph: float,
        class_name: str,
        strategy: str
    ) -> List[Dict]:
        """Create pairs when both gases are available"""
        pairs = []

        if strategy == 'random':
            # Randomly pair CO2 and CH4 samples
            min_len = min(len(co2_samples), len(ch4_samples))
            co2_subset = co2_samples.sample(n=min_len, random_state=42).reset_index(drop=True)
            ch4_subset = ch4_samples.sample(n=min_len, random_state=42).reset_index(drop=True)

            for i in range(min_len):
                pairs.append({
                    'co2_frame': co2_subset.iloc[i]['frame_path'],
                    'co2_mask': co2_subset.iloc[i]['mask_path'],
                    'ch4_frame': ch4_subset.iloc[i]['frame_path'],
                    'ch4_mask': ch4_subset.iloc[i]['mask_path'],
                    'ph_value': ph,
                    'class_id': int(co2_subset.iloc[i]['class_id']),
                    'class_name': class_name,
                    'is_paired': True,
                    'missing_modality': None
                })

            # Add remaining unpaired samples
            if len(co2_samples) > min_len:
                remaining_co2 = co2_samples.iloc[min_len:]
                pairs.extend(self._create_unpaired_samples(
                    remaining_co2, ph, class_name, missing='ch4'
                ))
            if len(ch4_samples) > min_len:
                remaining_ch4 = ch4_samples.iloc[min_len:]
                pairs.extend(self._create_unpaired_samples(
                    remaining_ch4, ph, class_name, missing='co2'
                ))

        elif strategy == 'original_based':
            # Pair based on original sample ID (if augmentations from same original)
            co2_grouped = co2_samples.groupby('original_sample_id')
            ch4_grouped = ch4_samples.groupby('original_sample_id')

            common_ids = set(co2_grouped.groups.keys()) & set(ch4_grouped.groups.keys())

            for orig_id in common_ids:
                co2_group = co2_grouped.get_group(orig_id)
                ch4_group = ch4_grouped.get_group(orig_id)

                # Pair by augmentation_id
                for _, co2_row in co2_group.iterrows():
                    aug_id = co2_row['augmentation_id']
                    ch4_match = ch4_group[ch4_group['augmentation_id'] == aug_id]

                    if len(ch4_match) > 0:
                        ch4_row = ch4_match.iloc[0]
                        pairs.append({
                            'co2_frame': co2_row['frame_path'],
                            'co2_mask': co2_row['mask_path'],
                            'ch4_frame': ch4_row['frame_path'],
                            'ch4_mask': ch4_row['mask_path'],
                            'ph_value': ph,
                            'class_id': int(co2_row['class_id']),
                            'class_name': class_name,
                            'is_paired': True,
                            'missing_modality': None
                        })

        return pairs

    def _create_unpaired_samples(
        self,
        samples: pd.DataFrame,
        ph: float,
        class_name: str,
        missing: str
    ) -> List[Dict]:
        """Create unpaired samples (one gas missing)"""
        pairs = []
        gas_type = samples.iloc[0]['gas_type']

        for _, row in samples.iterrows():
            if gas_type == 'co2':
                pairs.append({
                    'co2_frame': row['frame_path'],
                    'co2_mask': row['mask_path'],
                    'ch4_frame': None,  # Will be zero-padded
                    'ch4_mask': None,
                    'ph_value': ph,
                    'class_id': int(row['class_id']),
                    'class_name': class_name,
                    'is_paired': False,
                    'missing_modality': 'ch4'
                })
            else:  # ch4
                pairs.append({
                    'co2_frame': None,
                    'co2_mask': None,
                    'ch4_frame': row['frame_path'],
                    'ch4_mask': row['mask_path'],
                    'ph_value': ph,
                    'class_id': int(row['class_id']),
                    'class_name': class_name,
                    'is_paired': False,
                    'missing_modality': 'co2'
                })

        return pairs

    def get_pairing_statistics(self, pairs: List[Dict]) -> Dict:
        """Get statistics about pairing"""
        stats = {
            'total': len(pairs),
            'fully_paired': sum(1 for p in pairs if p['is_paired']),
            'co2_only': sum(1 for p in pairs if p['missing_modality'] == 'ch4'),
            'ch4_only': sum(1 for p in pairs if p['missing_modality'] == 'co2'),
            'by_class': {},
            'by_ph': {}
        }

        # Group by class
        for class_name in ['Healthy', 'Transitional', 'Acidotic']:
            class_pairs = [p for p in pairs if p['class_name'] == class_name]
            stats['by_class'][class_name] = {
                'total': len(class_pairs),
                'paired': sum(1 for p in class_pairs if p['is_paired'])
            }

        # Group by pH
        for ph in [5.0, 5.3, 5.6, 5.9, 6.2, 6.5]:
            ph_pairs = [p for p in pairs if p['ph_value'] == ph]
            stats['by_ph'][ph] = {
                'total': len(ph_pairs),
                'paired': sum(1 for p in ph_pairs if p['is_paired'])
            }

        return stats

    def save_paired_annotations(self, pairs: List[Dict], output_path: str):
        """Save paired annotations to CSV"""
        df = pd.DataFrame(pairs)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved paired annotations to {output_path}")


if __name__ == "__main__":
    # Test pairing
    dataset_root = "../../dataset"  # Acidosis/dataset
    pairer = GasPairCreator(dataset_root)

    for split in ['train', 'val', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print('='*50)

        df = pairer.load_annotations(split)
        pairs = pairer.create_pairs(df, pairing_strategy='random')
        stats = pairer.get_pairing_statistics(pairs)

        print(f"\nPairing Statistics:")
        print(f"  Total pairs: {stats['total']}")
        print(f"  Fully paired: {stats['fully_paired']} ({stats['fully_paired']/stats['total']*100:.1f}%)")
        print(f"  CO2 only: {stats['co2_only']}")
        print(f"  CH4 only: {stats['ch4_only']}")

        print(f"\nBy Class:")
        for class_name, class_stats in stats['by_class'].items():
            if class_stats['total'] > 0:
                print(f"  {class_name}: {class_stats['total']} total, {class_stats['paired']} paired")

        print(f"\nBy pH:")
        for ph, ph_stats in stats['by_ph'].items():
            if ph_stats['total'] > 0:
                print(f"  pH {ph}: {ph_stats['total']} total, {ph_stats['paired']} paired")

        # Save paired annotations
        output_path = f"paired_{split}_annotations.csv"
        pairer.save_paired_annotations(pairs, output_path)
