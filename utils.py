import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Standalone GTA benchmark evaluation with step-by-step mode')
    parser.add_argument('--model', type=str, default='qwen3-8b',
                       help='Model name to evaluate')
    parser.add_argument('--config', type=str, default='llama_cpp_server_qwen8b.json',
                       help='Server configuration file')
    parser.add_argument('--dataset_path', type=str, default='data/gta_dataset',
                       help='Path to GTA dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/standalone_gta',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--max_turns', type=int, default=10,
                       help='Maximum number of conversation turns')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for model generation')
    parser.add_argument('--save_steps', type=int, default=45,
                       help='Save steps')
    parser.add_argument('--step_by_step', action='store_true', default=True,
                       help='Enable step-by-step evaluation with ground truth tool outputs')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of samples to evaluate')
    parser.add_argument('--standalone', action='store_true',
                       help='Run standalone evaluation on existing results file')
    parser.add_argument('--results_file', type=str,
                       help='Path to existing evaluation_results.json for standalone evaluation')
    return parser.parse_args()