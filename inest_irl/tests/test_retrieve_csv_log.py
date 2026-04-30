import argparse

from inest_irl.utils.loggers import CSVLogger


def main(args):
    logger = CSVLogger(args.csv_path)
    eval_loss = logger.get_field_data(args.field, data_type=float)
    print(f"Retrieved loss data: {eval_loss}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CSVLogger data retrieval.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV log file.")
    parser.add_argument("--field", type=str, default="loss", help="Field name to retrieve from the CSV log.")
    args = parser.parse_args()
    
    main(args)