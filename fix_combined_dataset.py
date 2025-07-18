"""
Fix combined dataset generation by creating separate files for each domain
and a master index file.
"""

import csv
import json

def create_combined_csv():
    """Create a simplified combined CSV with essential fields only."""
    
    # Read individual domain CSVs
    domains = ["medical", "financial", "ecommerce"]
    combined_data = []
    
    for domain in domains:
        try:
            # Read training data
            with open(f"/home/ubuntu/{domain}_training_data.csv", 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract essential fields
                    essential_row = {
                        "domain": domain,
                        "outcome": row["outcome"],
                        "data_type": "training"
                    }
                    
                    # Add domain-specific key fields
                    if domain == "medical":
                        essential_row.update({
                            "age": row.get("age", ""),
                            "gender": row.get("gender", ""),
                            "symptom_count": row.get("symptom_count", ""),
                            "bp_systolic": row.get("bp_systolic", ""),
                            "heart_rate": row.get("heart_rate", "")
                        })
                    elif domain == "financial":
                        essential_row.update({
                            "age": row.get("age", ""),
                            "income": row.get("income", ""),
                            "risk_tolerance": row.get("risk_tolerance", ""),
                            "time_horizon": row.get("time_horizon", ""),
                            "stocks_percentage": row.get("stocks_percentage", "")
                        })
                    elif domain == "ecommerce":
                        essential_row.update({
                            "age": row.get("age", ""),
                            "gender": row.get("gender", ""),
                            "income_bracket": row.get("income_bracket", ""),
                            "purchase_history_count": row.get("purchase_history_count", ""),
                            "browsing_count": row.get("browsing_count", "")
                        })
                    
                    combined_data.append(essential_row)
            
            # Read test data
            with open(f"/home/ubuntu/{domain}_test_data.csv", 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract essential fields
                    essential_row = {
                        "domain": domain,
                        "outcome": row["outcome"],
                        "data_type": "test"
                    }
                    
                    # Add domain-specific key fields (same logic as above)
                    if domain == "medical":
                        essential_row.update({
                            "age": row.get("age", ""),
                            "gender": row.get("gender", ""),
                            "symptom_count": row.get("symptom_count", ""),
                            "bp_systolic": row.get("bp_systolic", ""),
                            "heart_rate": row.get("heart_rate", "")
                        })
                    elif domain == "financial":
                        essential_row.update({
                            "age": row.get("age", ""),
                            "income": row.get("income", ""),
                            "risk_tolerance": row.get("risk_tolerance", ""),
                            "time_horizon": row.get("time_horizon", ""),
                            "stocks_percentage": row.get("stocks_percentage", "")
                        })
                    elif domain == "ecommerce":
                        essential_row.update({
                            "age": row.get("age", ""),
                            "gender": row.get("gender", ""),
                            "income_bracket": row.get("income_bracket", ""),
                            "purchase_history_count": row.get("purchase_history_count", ""),
                            "browsing_count": row.get("browsing_count", "")
                        })
                    
                    combined_data.append(essential_row)
                    
        except FileNotFoundError:
            print(f"Warning: {domain} CSV files not found")
    
    # Write combined CSV
    if combined_data:
        with open("/home/ubuntu/combined_dataset.csv", 'w', newline='') as f:
            fieldnames = combined_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_data)
        
        print(f"Created combined_dataset.csv with {len(combined_data)} samples")
    
    return combined_data

def create_dataset_summary():
    """Create a summary of all generated datasets."""
    
    summary = {
        "datasets": {},
        "total_samples": 0,
        "domains": ["medical", "financial", "ecommerce"]
    }
    
    for domain in summary["domains"]:
        domain_summary = {
            "training_samples": 0,
            "test_samples": 0,
            "files": []
        }
        
        # Count training samples
        try:
            with open(f"/home/ubuntu/{domain}_training_data.csv", 'r') as f:
                reader = csv.DictReader(f)
                domain_summary["training_samples"] = sum(1 for _ in reader)
                domain_summary["files"].append(f"{domain}_training_data.csv")
                domain_summary["files"].append(f"{domain}_training_data.py")
        except FileNotFoundError:
            pass
        
        # Count test samples
        try:
            with open(f"/home/ubuntu/{domain}_test_data.csv", 'r') as f:
                reader = csv.DictReader(f)
                domain_summary["test_samples"] = sum(1 for _ in reader)
                domain_summary["files"].append(f"{domain}_test_data.csv")
        except FileNotFoundError:
            pass
        
        summary["datasets"][domain] = domain_summary
        summary["total_samples"] += domain_summary["training_samples"] + domain_summary["test_samples"]
    
    # Save summary
    with open("/home/ubuntu/dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

if __name__ == "__main__":
    print("Fixing combined dataset generation...")
    
    # Create combined CSV
    combined_data = create_combined_csv()
    
    # Create summary
    summary = create_dataset_summary()
    
    print("\nDataset Summary:")
    print(f"Total samples across all domains: {summary['total_samples']}")
    
    for domain, info in summary["datasets"].items():
        print(f"\n{domain.title()} Domain:")
        print(f"  Training samples: {info['training_samples']}")
        print(f"  Test samples: {info['test_samples']}")
        print(f"  Files: {', '.join(info['files'])}")
    
    print(f"\nAdditional files:")
    print(f"  - combined_dataset.csv ({len(combined_data)} samples)")
    print(f"  - dataset_summary.json")
    
    print("\n✅ Dataset generation completed successfully!")

