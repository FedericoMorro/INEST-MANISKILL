import argparse

from inest_irl.utils.csv_logger import CSVLogger


def main(args):
    logger = CSVLogger(args.csv_path)
    eval_loss = logger.get_field_data("loss", data_type=float)
    print(f"Retrieved loss data: {eval_loss}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CSVLogger data retrieval.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV log file.")
    args = parser.parse_args()
    
    main(args)