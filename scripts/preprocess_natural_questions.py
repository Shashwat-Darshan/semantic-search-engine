import os
import json
import time
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & CLI ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a very large Natural Questions JSONL (~20GB) in chunks with checkpointing."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the `simplified-nq-train.jsonl` file."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed/natural_questions",
        help="Directory to save processed chunk files and final output."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100_000,
        help="Number of lines to process per chunk. Adjust based on available RAM."
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default="nq_preprocess.checkpoint",
        help="Path to a JSON checkpoint file that tracks completed chunks."
    )
    parser.add_argument(
        "--use_gpu", action="store_true",
        help="If set, print GPU name & memory usage (parsing remains CPU-based)."
    )
    return parser.parse_args()


def configure_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHECKPOINT IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
class Checkpoint:
    def __init__(self, path):
        self.path = path
        self.completed = set()
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.completed = set(data.get("completed_chunks", []))
                logging.info(f"Loaded checkpoint: {len(self.completed)} chunks already completed.")
            except Exception as e:
                logging.error(f"Failed to load checkpoint file: {e}. Starting fresh.")
                self.completed = set()
                self._write()
        else:
            logging.info("No checkpoint file found. Starting from scratch.")
            self._write()

    def mark_done(self, chunk_id):
        """Mark a chunk ID as completed and write back to disk."""
        self.completed.add(chunk_id)
        self._write()

    def _write(self):
        """Write the checkpoint JSON to disk."""
        try:
            tmp = {"completed_chunks": sorted(list(self.completed))}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(tmp, f)
            logging.debug(f"Checkpoint updated: {len(self.completed)} chunks completed.")
        except Exception as e:
            logging.error(f"Unable to write checkpoint file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GPU INFORMATION (OPTIONAL)
# ─────────────────────────────────────────────────────────────────────────────
def show_gpu_info():
    if not GPU_AVAILABLE:
        logging.warning("GPU not available or PyTorch not installed. Parsing will run on CPU.")
        return
    try:
        gpu_name = torch.cuda.get_device_name(0)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        logging.info(f"🚀 GPU detected: {gpu_name}")
        logging.info(f"🧠 GPU Memory Free: {free_mem / 1e6:.2f} MB / Total: {total_mem / 1e6:.2f} MB")
    except Exception as e:
        logging.error(f"Failed to query GPU info: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. JSONL PARSING: Line-Level Function for Multiprocessing
# ─────────────────────────────────────────────────────────────────────────────
def parse_line(line: str):
    """
    Parse a single JSONL line from Natural Questions:
    - Extract 'question_text'
    - Use first 'short_answer' if exists; otherwise, use 'long_answer'
    - Return a dict {'question': ..., 'answer': ...} or None if invalid
    """
    try:
        item = json.loads(line)
        q = item.get("question_text", "").strip()
        if not q:
            return None

        shorts = item.get("short_answers", [])
        if shorts and shorts[0].get("text", "").strip():
            a = shorts[0]["text"].strip()
        else:
            a = item.get("long_answer", {}).get("text", "").strip()

        if not a:
            return None

        return {"question": q, "answer": a}
    except Exception:
        return None  # Skip malformed lines quietly


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHUNK PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def process_chunk(chunk_lines, chunk_id, output_dir):
    """
    Process one chunk of lines (list of raw JSONL lines):
    - Parse in parallel using multiprocessing Pool
    - Filter out None results
    - Save DataFrame to 'processed_chunk_{chunk_id}.csv' in output_dir
    - Return the saved file path
    """
    num_cores = cpu_count()
    logging.info(f"Processing chunk {chunk_id} with {len(chunk_lines)} lines using {num_cores} cores...")

    with Pool(num_cores) as pool:
        parsed = list(
            tqdm(
                pool.imap(parse_line, chunk_lines),
                total=len(chunk_lines),
                desc=f"Chunk {chunk_id}"
            )
        )

    cleaned = [row for row in parsed if row]
    df = pd.DataFrame(cleaned)

    chunk_file = os.path.join(output_dir, f"processed_chunk_{chunk_id}.csv")
    try:
        df.to_csv(chunk_file, index=False)
        logging.info(f"Saved chunk {chunk_id}: {len(df)} rows → {chunk_file}")
    except Exception as e:
        logging.error(f"Failed to save chunk {chunk_id}: {e}")
        raise

    return chunk_file


# ─────────────────────────────────────────────────────────────────────────────
# 6. STREAMING CHUNKS FROM LARGE JSONL
# ─────────────────────────────────────────────────────────────────────────────
def stream_chunks(input_path: str, chunk_size: int):
    """
    Generator that yields (chunk_id, list_of_lines) by reading `chunk_size` lines at a time.
    """
    chunk = []
    chunk_id = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            chunk.append(line)
            if idx % chunk_size == 0:
                yield chunk_id, chunk
                chunk = []
                chunk_id += 1

        # Final partial chunk
        if chunk:
            yield chunk_id, chunk


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN ORCHESTRATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    configure_logging()

    logging.info("=== Starting Natural Questions Preprocessing Pipeline ===")

    # 1. Show GPU info if requested
    if args.use_gpu:
        show_gpu_info()

    # 2. Initialize checkpoint
    chk = Checkpoint(args.checkpoint_file)

    start_time = time.time()
    total_chunks = 0
    processed_files = []

    # 3. Create output_dir if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Stream through file in chunks
    for chunk_id, lines in stream_chunks(args.input_path, args.chunk_size):
        total_chunks = max(total_chunks, chunk_id + 1)

        if chunk_id in chk.completed:
            logging.info(f"✓ Skipping chunk {chunk_id} (already completed).")
            continue

        try:
            chunk_file = process_chunk(lines, chunk_id, args.output_dir)
            processed_files.append(chunk_file)
            chk.mark_done(chunk_id)  # Update checkpoint immediately

        except Exception as e:
            logging.error(f"Chunk {chunk_id} processing failed: {e}. Aborting pipeline.")
            break  # Stop further processing

    # 5. Merge all processed chunks into one final CSV
    logging.info("🔗 Merging all processed chunk CSV files ...")
    chunk_filenames = sorted([
        fname for fname in os.listdir(args.output_dir)
        if fname.startswith("processed_chunk_") and fname.endswith(".csv")
    ])

    merge_start = time.time()
    df_list = []
    for fname in chunk_filenames:
        path = os.path.join(args.output_dir, fname)
        try:
            df_list.append(pd.read_csv(path))
        except Exception as e:
            logging.error(f"Failed to read chunk file {fname}: {e}")

    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_csv = os.path.join(args.output_dir, "processed_nq.csv")
        final_df.to_csv(final_csv, index=False)
        logging.info(f"Saved final merged dataset: {final_csv}")
    else:
        logging.warning("No chunk files found to merge.")

    merge_end = time.time()
    total_time = time.time() - start_time

    logging.info(f"=== Completed: {len(chk.completed)}/{total_chunks} chunks processed ===")
    logging.info(f"⌛ Total elapsed time: {total_time/60:.2f} minutes")
    logging.info(f"🔀 Merging time: {(merge_end - merge_start)/60:.2f} minutes")


if __name__ == "__main__":
    main()
