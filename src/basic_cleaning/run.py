#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Initialize a new W&B run
    logger.info("Initializing W&B run for basic cleaning...")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download the input artifact
    logger.info(f"Downloading input artifact: {args.input_artifact}...")
    local_path = run.use_artifact(args.input_artifact).file()
    logger.info(f"Input artifact downloaded to: {local_path}")

    # Load the data into a DataFrame
    logger.info("Loading data into DataFrame...")
    df = pd.read_csv(local_path)

    # Drop outliers based on price
    logger.info("Dropping outliers based on price...")
    min_price = args.min_price
    max_price = args.max_price
    logger.info(f"Filtering prices between {min_price} and {max_price}...")
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    logger.info(f"Data shape after dropping outliers: {df.shape}")

    # Convert 'last_review' column to datetime
    logger.info("Converting 'last_review' column to datetime...")
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("Conversion complete.")

    # Save the cleaned data to a new CSV file
    logger.info("Saving cleaned data to 'clean_sample.csv'...")
    df.to_csv("clean_sample.csv", index=False)
    logger.info("Cleaned data saved.")

    # Create a new W&B artifact and log the cleaned CSV file
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info(f"Logged artifact: {args.output_artifact}")

    # Finish the W&B run
    run.finish()
    logger.info("W&B run finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The W&B artifact with 'latest' tag to download for input, (e.g., 'sample.csv:latest')",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the W&B artifact to create for the cleaned data (e.g., 'clean_sample.csv')",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact (e.g., 'clean_data')",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Detailed description for the output artifact, (e.g., 'Data after removing outliers and parsing dates')",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price threshold for filtering data to remove outliers (e.g., 10.0)",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price threshold for filtering data to remove outliers (e.g., 350.0)",
        required=True
    )

    args = parser.parse_args()
    
    go(args)
