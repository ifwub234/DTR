import argparse
import json
import os
import glob

from evaluation_utils import cal_score


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VideoHallucer predictions from JSON files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing prediction JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Model name for output file naming",
    )
    parser.add_argument(
        "--eval_types",
        type=str,
        nargs="*",
        default=None,
        choices=["obj_rel", "temporal", "semantic", "interaction", "fact", "nonfact", "factdet"],
        help="Specific evaluation types to run. If not provided, evaluate all available types.",
    )
    args = parser.parse_args()

    # Define the mapping of file patterns to evaluation types
    eval_type_mapping = {
        "obj_rel": "obj_rel_predictions.json",
        "temporal": "temporal_predictions.json",
        "semantic": "semantic_predictions.json",
        "interaction": "interaction_predictions.json",
        "fact": "fact_predictions.json",
        "nonfact": "nonfact_predictions.json",
        "factdet": "factdet_predictions.json",
    }

    # Determine which types to evaluate
    if args.eval_types is None:
        eval_types = list(eval_type_mapping.keys())
    else:
        eval_types = args.eval_types

    # Collect all prediction files
    prediction_files = {}
    for eval_type in eval_types:
        pattern = os.path.join(args.input_dir, eval_type_mapping[eval_type])
        files = glob.glob(pattern)
        if files:
            # Use the first matching file for each type
            prediction_files[eval_type] = files[0]
            print(f"Found prediction file for {eval_type}: {files[0]}")
        else:
            print(f"Warning: No prediction file found for {eval_type}")

    if not prediction_files:
        print("Error: No prediction files found in the input directory.")
        return

    # Evaluate each type
    final_result = {}
    print("=" * 50)
    print(f"Evaluating predictions from: {args.input_dir}")
    print("=" * 50)

    for eval_type, file_path in prediction_files.items():
        print(f"\nEvaluating {eval_type}...")
        try:
            with open(file_path, "r") as f:
                predictions = json.load(f)

            scores = cal_score(predictions)
            final_result[eval_type] = scores

            print(f"  Basic Accuracy: {scores['basic_accuracy']:.4f}")
            print(f"  Hallucination Accuracy: {scores['halluc_accuracy']:.4f}")
            print(f"  Overall Accuracy: {scores['accuracy']:.4f}")
        except Exception as e:
            print(f"  Error evaluating {eval_type}: {str(e)}")
            continue

    # Calculate average scores across all evaluated types
    if final_result:
        final_acc = 0
        final_basic_acc = 0
        final_halluc_acc = 0

        for eval_type, result in final_result.items():
            final_basic_acc += result["basic_accuracy"]
            final_halluc_acc += result["halluc_accuracy"]
            final_acc += result["accuracy"]

        num_evaluated = len(final_result)
        final_acc = final_acc / num_evaluated
        final_basic_acc = final_basic_acc / num_evaluated
        final_halluc_acc = final_halluc_acc / num_evaluated

        final_result["all"] = {
            "basic_accuracy": final_basic_acc,
            "halluc_accuracy": final_halluc_acc,
            "accuracy": final_acc,
        }

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Save results
        output_file = os.path.join(args.output_dir, f"{args.model_name}_evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(final_result, f, indent=4)

        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Model: {args.model_name}")
        print(f"Number of evaluated types: {num_evaluated}")
        print(f"Average Basic Accuracy: {final_basic_acc:.4f}")
        print(f"Average Hallucination Accuracy: {final_halluc_acc:.4f}")
        print(f"Average Overall Accuracy: {final_acc:.4f}")
        print("=" * 50)

        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo successful evaluations completed.")


if __name__ == "__main__":
    main()
