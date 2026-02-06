"""
Report Generator Module

Generates structured validation reports including CSV summaries and statistics.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from validate.validation_result import ValidationResult, FailureReport

logger = logging.getLogger('validation.reports')


class ReportGenerator:
    """
    Generates structured validation reports.
    
    Creates:
    - validation_summary.csv (per-clip metrics)
    - failure_report.csv (flagged clips)
    - summary_statistics.json (aggregate stats)
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ReportGenerator: output_dir={output_dir}")
    
    def generate_summary_csv(self,
                            results_by_clip: Dict[str, List[ValidationResult]],
                            output_path: str):
        """
        Generate validation_summary.csv with per-clip metrics.
        
        Args:
            results_by_clip: Dictionary mapping clip_id to list of ValidationResults
            output_path: Path to save CSV
        """
        rows = []
        
        for clip_id, results in results_by_clip.items():
            row = {'clip_id': clip_id}
            
            # Overall status (worst status across all validators)
            statuses = [r.status for r in results]
            if 'FAIL' in statuses:
                row['status'] = 'FAIL'
            elif 'WARN' in statuses:
                row['status'] = 'WARN'
            else:
                row['status'] = 'PASS'
            
            # Extract metrics from each validator
            for result in results:
                validator_name = result.validator_name.replace('Validator', '').lower()
                
                for metric_name, metric_value in result.metrics.items():
                    col_name = f"{validator_name}_{metric_name}"
                    row[col_name] = metric_value
                
                # Add flags
                if result.flags:
                    row[f"{validator_name}_flags"] = ','.join(result.flags)
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Generated validation summary: {output_path}")
        logger.info(f"  Total clips: {len(df)}")
        logger.info(f"  PASS: {len(df[df['status'] == 'PASS'])}")
        logger.info(f"  WARN: {len(df[df['status'] == 'WARN'])}")
        logger.info(f"  FAIL: {len(df[df['status'] == 'FAIL'])}")
    
    def generate_failure_report(self,
                               failures: List[FailureReport],
                               output_path: str):
        """
        Generate failure_report.csv with flagged clips.
        
        Args:
            failures: List of FailureReport objects
            output_path: Path to save CSV
        """
        if len(failures) == 0:
            logger.info("No failures to report")
            return
        
        rows = []
        
        for failure in failures:
            row = {
                'clip_id': failure.clip_id,
                'failure_modes': ','.join(failure.failure_modes),
                'severity': failure.severity,
                'recommended_action': failure.recommended_action
            }
            
            # Add diagnostic info as separate columns
            for key, value in failure.diagnostic_info.items():
                row[f"diagnostic_{key}"] = value
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Generated failure report: {output_path}")
        logger.info(f"  Total failures: {len(df)}")
        logger.info(f"  HIGH severity: {len(df[df['severity'] == 'HIGH'])}")
        logger.info(f"  MEDIUM severity: {len(df[df['severity'] == 'MEDIUM'])}")
        logger.info(f"  LOW severity: {len(df[df['severity'] == 'LOW'])}")
    
    def generate_summary_statistics(self,
                                   results_by_clip: Dict[str, List[ValidationResult]]) -> Dict:
        """
        Compute aggregate statistics for the dataset.
        
        Args:
            results_by_clip: Dictionary mapping clip_id to list of ValidationResults
        
        Returns:
            Dictionary with summary statistics
        """
        total_clips = len(results_by_clip)
        
        # Count statuses
        status_counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
        
        for results in results_by_clip.values():
            statuses = [r.status for r in results]
            if 'FAIL' in statuses:
                status_counts['FAIL'] += 1
            elif 'WARN' in statuses:
                status_counts['WARN'] += 1
            else:
                status_counts['PASS'] += 1
        
        # Compute aggregate metrics
        all_metrics = {}
        
        for results in results_by_clip.values():
            for result in results:
                validator_name = result.validator_name
                
                if validator_name not in all_metrics:
                    all_metrics[validator_name] = {}
                
                for metric_name, metric_value in result.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in all_metrics[validator_name]:
                            all_metrics[validator_name][metric_name] = []
                        all_metrics[validator_name][metric_name].append(metric_value)
        
        # Compute mean/std for each metric
        aggregated_metrics = {}
        
        for validator_name, metrics in all_metrics.items():
            aggregated_metrics[validator_name] = {}
            
            for metric_name, values in metrics.items():
                import numpy as np
                aggregated_metrics[validator_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        summary = {
            'total_clips': total_clips,
            'status_counts': status_counts,
            'pass_rate': status_counts['PASS'] / total_clips if total_clips > 0 else 0,
            'aggregated_metrics': aggregated_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def save_summary_statistics(self, summary: Dict, output_path: str):
        """
        Save summary statistics to JSON file.
        
        Args:
            summary: Summary statistics dictionary
            output_path: Path to save JSON
        """
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary statistics: {output_path}")
