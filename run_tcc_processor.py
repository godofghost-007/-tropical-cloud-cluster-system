#!/usr/bin/env python3
"""
Runner script for TCC (Tropical Cloud Cluster) Processor
Handles data processing, detection, tracking, and risk analysis
"""

import os
import sys
import logging
import yaml
from datetime import datetime
from tcc_processor import TCCProcessor, CONFIG

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"tcc_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def load_config():
    """Load configuration from YAML file"""
    config_file = "tcc_config.yaml"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logging.error(f"Error loading config from {config_file}: {str(e)}")
            logging.info("Using default configuration")
            return CONFIG
    else:
        logging.warning(f"Config file {config_file} not found. Using default configuration.")
        # Save default config
        try:
            with open(config_file, 'w') as f:
                yaml.dump(CONFIG, f, default_flow_style=False)
            logging.info(f"Saved default configuration to {config_file}")
        except Exception as e:
            logging.error(f"Error saving default config: {str(e)}")
        return CONFIG

def check_data_files(config):
    """Check if data files exist in the input directory"""
    input_dir = config["data"]["input_dir"]
    
    if not os.path.exists(input_dir):
        logging.error(f"Input directory {input_dir} does not exist")
        return False
    
    # Look for supported file formats
    supported_extensions = ('.nc', '.nc4', '.hdf', '.h4', '.h5')
    data_files = [
        f for f in os.listdir(input_dir) 
        if f.lower().endswith(supported_extensions)
    ]
    
    if not data_files:
        logging.error(f"No supported data files found in {input_dir}")
        logging.info(f"Supported formats: {supported_extensions}")
        return False
    
    logging.info(f"Found {len(data_files)} data files in {input_dir}")
    return True

def create_directories(config):
    """Create necessary output directories"""
    output_dir = config["data"]["output_dir"]
    directories = [
        output_dir,
        os.path.join(output_dir, "tracks"),
        os.path.join(output_dir, "visualizations"),
        os.path.join(output_dir, "reports")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def main():
    """Main execution function"""
    print("🌪️ TCC (Tropical Cloud Cluster) Processor")
    print("=" * 50)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting TCC processing pipeline")
    
    try:
        # Load configuration
        config = load_config()
        
        # Check data files
        if not check_data_files(config):
            print("❌ No data files found. Please add INSAT-3D data files to the input directory.")
            return 1
        
        # Create output directories
        create_directories(config)
        
        # Get list of data files
        input_dir = config["data"]["input_dir"]
        supported_extensions = ('.nc', '.nc4', '.hdf', '.h4', '.h5')
        file_paths = sorted([
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith(supported_extensions)
        ])
        
        print(f"📁 Processing {len(file_paths)} data files...")
        logging.info(f"Processing {len(file_paths)} files")
        
        # Initialize processor
        processor = TCCProcessor(config)
        
        # Process files
        results = processor.process_files(file_paths)
        
        # Generate summary
        if not results.empty:
            print("\n✅ Processing completed successfully!")
            print(f"📊 Results Summary:")
            print(f"   - Total tracks detected: {results['track_id'].nunique()}")
            print(f"   - Total observations: {len(results)}")
            print(f"   - Date range: {results['datetime'].min()} to {results['datetime'].max()}")
            print(f"   - Average risk score: {results['cyclogenesis_risk'].mean():.3f}")
            
            # Save summary report
            summary_file = os.path.join(config["data"]["output_dir"], "processing_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("TCC Processing Summary\n")
                f.write("=" * 30 + "\n")
                f.write(f"Processing date: {datetime.now()}\n")
                f.write(f"Files processed: {len(file_paths)}\n")
                f.write(f"Total tracks: {results['track_id'].nunique()}\n")
                f.write(f"Total observations: {len(results)}\n")
                f.write(f"Date range: {results['datetime'].min()} to {results['datetime'].max()}\n")
                f.write(f"Average risk: {results['cyclogenesis_risk'].mean():.3f}\n")
                f.write(f"Max risk: {results['cyclogenesis_risk'].max():.3f}\n")
            
            print(f"📄 Summary saved to: {summary_file}")
            print(f"📋 Log file: {log_file}")
            print(f"🌐 Run 'streamlit run dashboard.py' to view results")
            
        else:
            print("⚠️ No results generated. Check log file for details.")
            print(f"📋 Log file: {log_file}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        logging.info("Processing interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        logging.error(f"Processing failed: {str(e)}", exc_info=True)
        print(f"📋 Check log file for details: {log_file}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 