"""
run_pipeline.py - Complete Processing Pipeline
Runs all steps in sequence with error handling
"""

import os
import sys
import time
import generate_time_series
import batch_processor
import tracking
import validation

def main():
    """Run full processing pipeline"""
    print("Starting Tropical Cloud Cluster Processing Pipeline")
    start_time = time.time()
    
    # Step 1: Generate data
    print("\n=== Generating Synthetic Data ===")
    try:
        generate_time_series.generate_time_series(overwrite=True)
        print("Data generation successful")
    except Exception as e:
        print(f"Data generation failed: {str(e)}")
        sys.exit(1)
    
    # Step 2: Process timesteps
    print("\n=== Processing Timesteps ===")
    try:
        batch_processor.process_directory("data/time_series", output_dir="outputs/batch", force=True)
        print("Timestep processing complete")
    except Exception as e:
        print(f"Timestep processing failed: {str(e)}")
        sys.exit(1)
    
    # Step 3: Run tracking
    print("\n=== Tracking Clusters ===")
    try:
        tracking.track_clusters()
        print("Tracking complete")
    except Exception as e:
        print(f"Tracking failed: {str(e)}")
        sys.exit(1)
    
    # Step 4: Validate
    print("\n=== Validation ===")
    try:
        validation.main()
        print("Validation complete")
    except Exception as e:
        print(f"Validation failed: {str(e)}")
    
    # Final status
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 