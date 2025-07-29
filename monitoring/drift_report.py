from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def generate_drift_report(reference_path, current_path, output_path):
    ref = pd.read_csv(reference_path)
    curr = pd.read_csv(current_path)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)
    report.save_html(output_path)

if __name__ == '__main__':
    generate_drift_report('data/processed/reference.csv', 'data/processed/current.csv', 'monitoring/drift_report.html')
