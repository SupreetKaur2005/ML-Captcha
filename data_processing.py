# import os
# import json
# import pandas as pd
# import argparse
# from pathlib import Path

# def create_dataset(raw_human_directory='data/raw/human/', 
#                   raw_bot_directory='data/raw/bot/',
#                   processed_directory='data/processed/'):
#     """
#     Process raw human and bot data to create a labeled dataset
    
#     Args:
#         raw_human_directory: Path to directory containing human data JSON files
#         raw_bot_directory: Path to directory containing bot data JSON files
#         processed_directory: Path to save the processed dataset
    
#     Returns:
#         DataFrame of the processed dataset
#     """
#     # Create processed directory using pathlib (more robust)
#     Path(processed_directory).mkdir(parents=True, exist_ok=True)
    
#     all_data = []
    
#     # Process human data
#     human_path = Path(raw_human_directory)
#     if human_path.exists():
#         for file_path in human_path.glob('*.json'):
#             try:
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)
#                     data['label'] = 'human'
#                     all_data.append(data)
#             except json.JSONDecodeError:
#                 print(f"Warning: Could not parse JSON in {file_path}")
#             except Exception as e:
#                 print(f"Error processing {file_path}: {str(e)}")
    
#     # Process bot data
#     bot_path = Path(raw_bot_directory)
#     if bot_path.exists():
#         for file_path in bot_path.glob('*.json'):
#             try:
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)
#                     data['label'] = 'bot'
#                     all_data.append(data)
#             except json.JSONDecodeError:
#                 print(f"Warning: Could not parse JSON in {file_path}")
#             except Exception as e:
#                 print(f"Error processing {file_path}: {str(e)}")
    
#     if not all_data:
#         print("Warning: No data was loaded. Check your input directories.")
#         return None
        
#     # Convert to DataFrame
#     df = pd.DataFrame(all_data)

#     # Stringify the 'mouse_movements' column if it exists
#     if 'mouse_movements' in df.columns:
#         df['mouse_movements'] = df['mouse_movements'].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)

#     # Save to CSV
#     output_path = Path(processed_directory) / 'dataset.csv'
#     df.to_csv(output_path, index=False)
    
#     print(f"Created dataset with {len(df)} records")
#     print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
#     return df

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Process raw data to create labeled dataset")
#     parser.add_argument("--human_dir", default="data/raw/human/", help="Directory with human data")
#     parser.add_argument("--bot_dir", default="data/raw/bot/", help="Directory with bot data")
#     parser.add_argument("--output_dir", default="data/processed/", help="Directory to save processed data")
#     args = parser.parse_args()
    
#     create_dataset(args.human_dir, args.bot_dir, args.output_dir)


import os
import json
import pandas as pd
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_json_structure(data, required_fields):
    """
    Validate that the JSON data contains all required fields.
    Args:
        data (dict): The JSON data to validate.
        required_fields (list): List of required field names.
    Returns:
        bool: True if valid, False otherwise.
    """
    for field in required_fields:
        if field not in data:
            return False
    return True


def create_dataset(raw_human_directory='data/raw/human/',
                   raw_bot_directory='data/raw/bot/',
                   processed_directory='data/processed/',
                   chunk_size=1000):
    """
    Process raw human and bot data to create a labeled dataset.

    Args:
        raw_human_directory: Path to directory containing human data JSON files.
        raw_bot_directory: Path to directory containing bot data JSON files.
        processed_directory: Path to save the processed dataset.
        chunk_size: Number of records to process per chunk to handle large datasets.

    Returns:
        DataFrame of the processed dataset.
    """
    # Create processed directory using pathlib (more robust)
    Path(processed_directory).mkdir(parents=True, exist_ok=True)

    all_data = []
    required_fields = ['mouseEvents', 'keyEvents', 'clickEvents']

    # Process human data
    human_path = Path(raw_human_directory)
    if human_path.exists():
        for file_path in human_path.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if validate_json_structure(data, required_fields):
                        data['label'] = 'human'
                        all_data.append(data)
                    else:
                        logging.warning(f"Missing required fields in {file_path}. Skipping.")
            except json.JSONDecodeError:
                logging.warning(f"Could not parse JSON in {file_path}. Skipping.")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")

    # Process bot data
    bot_path = Path(raw_bot_directory)
    if bot_path.exists():
        for file_path in bot_path.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if validate_json_structure(data, required_fields):
                        data['label'] = 'bot'
                        all_data.append(data)
                    else:
                        logging.warning(f"Missing required fields in {file_path}. Skipping.")
            except json.JSONDecodeError:
                logging.warning(f"Could not parse JSON in {file_path}. Skipping.")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")

    if not all_data:
        logging.warning("No data was loaded. Check your input directories.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Convert non-hashable columns (lists/dictionaries) to strings
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(json.dumps)

    # Deduplicate the dataset
    before_dedup = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    logging.info(f"Removed {before_dedup - after_dedup} duplicate entries.")

    # Save in chunks for scalability
    output_path = Path(processed_directory) / 'dataset.csv'
    if chunk_size and len(df) > chunk_size:
        logging.info(f"Dataset is large. Saving in chunks of {chunk_size} records.")
        for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
            chunk_path = Path(processed_directory) / f'dataset_chunk_{i + 1}.csv'
            df.iloc[chunk_start:chunk_start + chunk_size].to_csv(chunk_path, index=False)
            logging.info(f"Saved chunk {i + 1} to {chunk_path}.")
    else:
        df.to_csv(output_path, index=False)
        logging.info(f"Created dataset with {len(df)} records")
        logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process raw data to create a labeled dataset.")
    parser.add_argument("--human_dir", default="data/raw/human/", help="Directory with human data JSON files.")
    parser.add_argument("--bot_dir", default="data/raw/bot/", help="Directory with bot data JSON files.")
    parser.add_argument("--output_dir", default="data/processed/", help="Directory to save the processed data.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of records per chunk for large datasets.")
    args = parser.parse_args()

    create_dataset(args.human_dir, args.bot_dir, args.output_dir, args.chunk_size)