#!/usr/bin/env python3
"""
Standalone GTA benchmark evaluation script with step-by-step evaluation.
This script evaluates LLM performance by feeding correct tool outputs from ground truth at each step.

Usage:
    python evaluation.py --model qwen3-8b --config llama_cpp_server_qwen8b.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
import copy
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel

from models import GTAModel, GTAAgent, create_model_from_config
from dataset import GTADataset, GTAEvaluator
from utils import parse_args

class JudgeResponse(BaseModel):
    is_correct: bool
    reason: str

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('standalone_gta_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_output_dir(config_file: str) -> str:
    """Create timestamped output directory using model_alias from config."""
    # Read model_alias from config file
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            model_alias = config.get('models')[0].get('model_alias')
            if not model_alias:
                raise ValueError(f"No model_alias found in config file: {config_file}")
    except Exception as e:
        logger.error(f"Failed to read model_alias from config: {e}")
        raise
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('outputs', f'{timestamp}_{model_alias}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_dataset(dataset_path: str, max_samples: int = None) -> GTADataset:
    """Load the GTA benchmark dataset."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        dataset = GTADataset(dataset_path)
        
        if max_samples and max_samples < len(dataset):
            # Limit dataset for testing
            dataset.data = dataset.data[:max_samples]
            logger.info(f"Limited dataset to {len(dataset)} samples")
        else:
            logger.info(f"Loaded {len(dataset)} samples")
            
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def create_model(config_file: str, model_name: str, temperature: float) -> GTAModel:
    """Create a model instance from configuration."""
    logger.info(f"Creating model {model_name} from config {config_file}")
    
    try:
        model = create_model_from_config(config_file)
        model.model_name = model_name
        model.temperature = temperature
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise


def extract_assistant_steps(dialogs: List[Dict]) -> List[Dict]:
    """Extract only assistant steps from the dialog (excluding user and tool messages)."""
    assistant_steps = []
    for msg in dialogs:
        if msg.get('role') != 'assistant':
            continue
        assistant_msg = copy.deepcopy(msg)
        if 'tool_calls' in assistant_msg:
            for tool_call in assistant_msg['tool_calls']:
                tool_call['function']['arguments'] = json.loads(tool_call['function']['arguments'])
        assistant_steps.append(assistant_msg)
    return assistant_steps


def build_chat_history_for_step(dialogs: List[Dict], step_idx: int) -> List[Dict]:
    """Build chat history up to a specific step for feeding to the LLM."""
    chat_history = []
    
    # Always start with the user query (first message)
    user_query = dialogs[0]
    chat_history.append(user_query)
    
    # Add assistant and tool messages up to the current step
    assistant_count = 0
    for i, msg in enumerate(dialogs[1:], 1):  # Skip the first user message
        if msg.get('role') == 'assistant':
            if assistant_count < step_idx:
                chat_history.append(msg)
                assistant_count += 1
            else:
                break
        elif msg.get('role') == 'tool' and assistant_count <= step_idx:
            # Add tool response if we've included the corresponding assistant message
            chat_history.append(msg)
    
    return chat_history


def run_step_by_step_evaluation(model: GTAModel, sample: Dict) -> Dict:
    """Run step-by-step evaluation on a single sample."""
    try:
        start_time = time.time()
        
        # Extract sample data
        user_query = sample['user_query']
        tools = sample['tools']
        files = sample['files']
        ground_truth_dialogs = sample['dialogs']
        reference = sample['gt_answer']
        
        if not user_query:
            logger.warning(f"No user query found in sample {sample['idx']}")
            return {
                'idx': sample['idx'],
                'gold': [],
                'prediction': [],
                'origin_prompt': [],
                'reference': reference,
                'error': 'No user query found',
                'duration': 0,
                'total_steps': 0,
                'successful_steps': 0
            }
        
        # Extract ground truth assistant steps
        gold_steps = extract_assistant_steps(ground_truth_dialogs)
        
        predictions = []
        origin_prompts = []
        successful_steps = 0
        
        # Create agent for single-step predictions
        agent = GTAAgent(model, max_turns=1)  # Only one turn per step
        
        # Run evaluation for each step
        for step_idx in range(len(gold_steps)):
            # logger.info(f"Step {step_idx + 1}/{len(gold_steps)}")
            try:
                # Build chat history for this step
                chat_history = build_chat_history_for_step(ground_truth_dialogs, step_idx)
                origin_prompts.append(chat_history.copy())
                
                # Get prediction for this step
                # We need to modify the agent to accept pre-built chat history
                prediction = agent.predict_single_step(chat_history, tools, files)
                predictions.append(prediction)
                successful_steps += 1
                
                logger.debug(f"Step {step_idx + 1}/{len(gold_steps)} completed for sample {sample['idx']}")
                
            except Exception as e:
                logger.error(f"Error in step {step_idx + 1} for sample {sample['idx']}: {e}")
                predictions.append({
                    'role': 'assistant',
                    'error': {
                        'type': 'STEP_ERROR',
                        'msg': str(e)
                    }
                })
                origin_prompts.append(chat_history.copy() if 'chat_history' in locals() else [])
        
        duration = time.time() - start_time
        
        return {
            'idx': sample['idx'],
            'gold': gold_steps,
            'prediction': predictions,
            'origin_prompt': origin_prompts,
            'reference': reference,
            'tools_available': [tool['name'] for tool in tools],
            'files_available': len(files),
            'duration': duration,
            'user_query': user_query,
            'total_steps': len(gold_steps),
            'successful_steps': successful_steps
        }
        
    except Exception as e:
        logger.error(f"Error in step-by-step evaluation for sample {sample.get('idx', 'unknown')}: {e}")
        return {
            'idx': sample.get('idx', -1),
            'gold': [],
            'prediction': [],
            'origin_prompt': [],
            'reference': sample.get('gt_answer', None),
            'error': str(e),
            'duration': 0,
            'total_steps': 0,
            'successful_steps': 0
        }


def run_single_evaluation(agent: GTAAgent, sample: Dict) -> Dict:
    """Run evaluation on a single sample (original method for backward compatibility)."""
    try:
        start_time = time.time()
        
        # Extract sample data
        user_query = sample['user_query']
        tools = sample['tools']
        files = sample['files']
        ground_truth = sample['dialogs']
        reference = sample['gt_answer']
        
        if not user_query:
            logger.warning(f"No user query found in sample {sample['idx']}")
            return {
                'idx': sample['idx'],
                'prediction': [],
                'ground_truth': ground_truth,
                'reference': reference,
                'error': 'No user query found',
                'duration': 0
            }
        
        # Run conversation
        prediction = agent.chat(user_query, tools, files)
        
        duration = time.time() - start_time
        
        return {
            'idx': sample['idx'],
            'prediction': prediction,
            'ground_truth': ground_truth,
            'reference': reference,
            'tools_available': [tool['name'] for tool in tools],
            'files_available': len(files),
            'duration': duration,
            'user_query': user_query
        }
        
    except Exception as e:
        logger.error(f"Error in single evaluation for sample {sample.get('idx', 'unknown')}: {e}")
        return {
            'idx': sample.get('idx', -1),
            'prediction': [],
            'ground_truth': sample.get('dialogs', []),
            'reference': sample.get('gt_answer', None),
            'error': str(e),
            'duration': 0
        }


def run_evaluation(model: GTAModel, dataset: GTADataset, 
                  output_dir: str, max_turns: int = 10, save_steps: int = 45,
                  step_by_step: bool = True, n_samples: int = 10) -> tuple:
    """Run evaluation on the entire dataset."""
    logger.info(f"Starting {'step-by-step' if step_by_step else 'standard'} evaluation on {len(dataset)} samples")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_duration = 0
    successful_samples = 0
    total_steps = 0
    successful_steps = 0
    
    for idx, sample in tqdm(enumerate(dataset[:n_samples]), total=min(n_samples, len(dataset))):
        # logger.info(f"Processing sample {idx + 1}/{len(dataset)} (ID: {sample['idx']})")
        
        if step_by_step:
            result = run_step_by_step_evaluation(model, sample)
            total_steps += result.get('total_steps', 0)
            successful_steps += result.get('successful_steps', 0)
        else:
            # Create agent for standard evaluation
            agent = GTAAgent(model, max_turns=max_turns)
            result = run_single_evaluation(agent, sample)
        
        results.append(result)
        
        if 'error' not in result:
            successful_samples += 1
            total_duration += result['duration']
        
        # Save intermediate results
        if (idx + 1) % save_steps == 0:
            intermediate_file = os.path.join(output_dir, f'intermediate_results_{idx + 1}.json')
            with open(intermediate_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved intermediate results to {intermediate_file}")
    
    # Save final results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved final results to {results_file}")
    
    # Log statistics
    avg_duration = total_duration / successful_samples if successful_samples > 0 else 0
    logger.info(f"Evaluation completed: {successful_samples}/{len(dataset)} samples successful")
    logger.info(f"Average duration per sample: {avg_duration:.2f} seconds")
    
    if step_by_step:
        step_success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        logger.info(f"Step-by-step stats: {successful_steps}/{total_steps} steps successful ({step_success_rate:.1f}%)")
    
    return results


def calculate_step_by_step_metrics(results: List[Dict], output_dir: str) -> Dict:
    """Calculate evaluation metrics for step-by-step evaluation."""
    logger.info("Calculating step-by-step evaluation metrics")
    
    try:
        # Initialize counters for step-by-step metrics
        total_samples = len(results)
        total_steps = 0
        successful_steps = 0
        
        # GTA benchmark specific metrics
        total_tool_steps = 0
        total_answer_steps = 0
        inst_align_count = 0
        tool_acc_count = 0
        args_acc_count = 0
        answer_acc_count = 0
        tool_call_count = 0
        tool_call_error_count = 0
        
        # Prepare data for traditional evaluator as backup
        all_predictions = []
        all_ground_truths = []
        all_references = []
        all_incorrect_answers = []
        
        for result in results:
            if 'error' in result:
                all_predictions.append([])
                all_ground_truths.append([])
                all_references.append(result.get('reference'))
                all_incorrect_answers.append([])
                continue
                
            gold_steps = result.get('gold', [])
            pred_steps = result.get('prediction', [])
            reference = result.get('reference')
            
            total_steps += len(gold_steps)
            
            # Pad shorter sequence with empty steps for comparison
            max_len = max(len(gold_steps), len(pred_steps))
            padded_gold = gold_steps + [{}] * (max_len - len(gold_steps))
            padded_pred = pred_steps + [{}] * (max_len - len(pred_steps))
            
            # Step-by-step comparison
            for i, (gold_step, pred_step) in enumerate(zip(padded_gold, padded_pred)):
                if not gold_step:  # Skip padding
                    continue
                    
                # Determine step types
                gold_type = _get_step_type(gold_step)
                pred_type = _get_step_type(pred_step) if pred_step else 'error'
                
                # Count step types
                if gold_type == 'tool':
                    total_tool_steps += 1
                elif gold_type == 'answer':
                    total_answer_steps += 1
                
                # Count prediction types
                if pred_type == 'tool':
                    tool_call_count += 1
                    if 'error' in pred_step:
                        tool_call_error_count += 1
                
                # Check if step was successful (no error)
                if 'error' not in pred_step and pred_step:
                    successful_steps += 1
                    
                    # Instruction alignment: correct step type
                    if pred_type == gold_type:
                        inst_align_count += 1
                        
                        # Tool accuracy: correct tool name
                        if gold_type == 'tool':
                            gold_tool_name = gold_step['tool_calls'][0]['function']['name']
                            pred_tool_name = pred_step['tool_calls'][0]['function']['name']
                            
                            if gold_tool_name == pred_tool_name:
                                tool_acc_count += 1
                                
                                # Arguments accuracy: correct arguments
                                gold_args = gold_step['tool_calls'][0]['function']['arguments']
                                pred_args = pred_step['tool_calls'][0]['function']['arguments']
                                
                                # Normalize arguments for comparison
                                if _normalize_args(gold_args) == _normalize_args(pred_args):
                                    args_acc_count += 1
                        
                        # Answer accuracy: correct final answer
                        elif gold_type == 'answer' and reference and i == len(gold_steps) - 1:
                            # pred_content = pred_step.get('content', '')
                            # if isinstance(reference, dict):
                            #     # Use whitelist/blacklist evaluation
                            #     if _check_answer_correctness_dict(pred_content, reference):
                            #         answer_acc_count += 1
                            #     else:
                            #         all_incorrect_answers.append({
                            #             "prompt": result.get('origin_prompt', [])[-1],
                            #             "pred": pred_content,
                            #             "gt": reference
                            #         })
                            # elif isinstance(reference, list):
                            #     # Use similarity scoring
                            #     answer_acc_count += _calculate_similarity_score(pred_content, reference)
                            #     all_incorrect_answers.append({
                            #         "prompt": result.get('origin_prompt', [])[-1],
                            #         "pred": pred_content,
                            #         "gt": reference
                            #     })
                            answer_acc_count += check_answer_correctness_llm(pred_step.get('content', ''), reference)

            
            # Add to traditional evaluator data
            all_predictions.append(pred_steps)
            all_ground_truths.append(gold_steps)
            all_references.append(reference)
        
        # Calculate final metrics
        metrics = {
            'total_samples': total_samples,
            'successful_samples': len([r for r in results if 'error' not in r]),
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'step_success_rate(inst_acc)': (successful_steps / total_steps * 100) if total_steps > 0 else 0,
            
            # GTA benchmark metrics
            'type_acc': (inst_align_count / total_steps * 100) if total_steps > 0 else 0,
            'tool_acc': (tool_acc_count / total_tool_steps * 100) if total_tool_steps > 0 else 0,
            'args_acc': (args_acc_count / total_tool_steps * 100) if total_tool_steps > 0 else 0,
            'answer_acc': (answer_acc_count / total_answer_steps * 100) if total_answer_steps > 0 else 0,
            
            # Additional statistics
            'total_tool_steps': total_tool_steps,
            'total_answer_steps': total_answer_steps,
            'tool_call_count': tool_call_count,
            'tool_call_error_count': tool_call_error_count,
            'sample_success_rate': (len([r for r in results if 'error' not in r]) / total_samples * 100) if total_samples > 0 else 0
        }
        
        # Try to use the original evaluator for additional metrics as backup
        # Deprecated. They are all overlapping.
        # try:
        #     evaluator = GTAEvaluator(mode='every_with_gt')
        #     additional_metrics = evaluator.evaluate(all_predictions, all_ground_truths, all_references)
        #     # Only add metrics that we haven't calculated ourselves
        #     for key, value in additional_metrics.items():
        #         if key not in metrics:
        #             metrics[key] = value
        # except Exception as e:
        #     logger.warning(f"Could not calculate additional metrics with GTAEvaluator: {e}")
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'step_by_step_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Step-by-step Evaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save incorrect answers
        incorrect_answers_file = os.path.join(output_dir, 'incorrect_answers.json')
        with open(incorrect_answers_file, 'w') as f:
            json.dump(all_incorrect_answers, f, indent=2)
        logger.info(f"Saved incorrect answers to {incorrect_answers_file}")

        return metrics
    
    except Exception as e:
        logger.error(f"Critical Error calculating step-by-step metrics: {e}")
        raise e


def _get_step_type(step: Dict) -> str:
    """Determine the type of a step (tool, answer, or error)."""
    if 'error' in step:
        return 'error'
    elif 'tool_calls' in step and step['tool_calls']:
        return 'tool'
    elif step.get('role') == 'assistant' and 'content' in step:
        return 'answer'
    else:
        return 'unknown'


def _normalize_args(args) -> str:
    """Normalize arguments for comparison."""
    if isinstance(args, str):
        try:
            # Try to parse as JSON and re-serialize for consistent formatting
            import json
            parsed = json.loads(args)
            return json.dumps(parsed, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return str(args).strip()
    elif isinstance(args, dict):
        import json
        return json.dumps(args, sort_keys=True)
    else:
        return str(args).strip()


def _check_answer_correctness_dict(pred_content: str, reference: Dict) -> bool:
    """Check if prediction matches reference using whitelist/blacklist."""
    import re
    pred_content = pred_content.replace(',', '')
    
    if not isinstance(reference, dict) or 'whitelist' not in reference:
        return False
        
    count = 0
    for aliases in reference['whitelist']:
        pattern = r'\b(?:' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
        if re.search(pattern, pred_content, re.IGNORECASE):
            count += 1
            
    if not reference.get('blacklist'):
        if count == len(reference['whitelist']):
            return True
    else:
        pattern_bk = r'\b(?:' + '|'.join(re.escape(alias) for aliases in reference['blacklist'] for alias in aliases) + r')\b'
        if count == len(reference['whitelist']) and not re.search(pattern_bk, pred_content, re.IGNORECASE):
            return True
    return False


def _calculate_similarity_score(pred_content: str, reference_list: List[str]) -> float:
    """Calculate similarity score using sentence transformers if available."""
    try:
        from sentence_transformers import SentenceTransformer, util
        import numpy as np
        
        model = SentenceTransformer('all-mpnet-base-v2')
        pred_emb = model.encode(pred_content, convert_to_tensor=True)
        
        max_score = 0.0
        for ref_text in reference_list:
            ref_emb = model.encode(ref_text, convert_to_tensor=True)
            score = np.maximum(util.cos_sim(pred_emb, ref_emb).cpu().numpy(), 0)
            if score[0][0] > max_score:
                max_score = score[0][0]
        
        return max_score
    except ImportError:
        # Fallback to simple string matching
        pred_lower = pred_content.lower()
        for ref_text in reference_list:
            if ref_text.lower() in pred_lower or pred_lower in ref_text.lower():
                return 1.0
        return 0.0


def calculate_metrics(predictions: List, ground_truths: List, references: List, 
                     output_dir: str) -> Dict:
    """Calculate evaluation metrics (original method for backward compatibility)."""
    logger.info("Calculating evaluation metrics")
    
    try:
        evaluator = GTAEvaluator(mode='every_with_gt')
        metrics = evaluator.evaluate(predictions, ground_truths, references)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Evaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}


def run_standalone_evaluation(results_file: str, output_dir: str) -> None:
    """Run standalone evaluation on existing results file."""
    logger.info(f"Running standalone evaluation on {results_file}")
    
    try:
        # Load existing results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Calculate metrics
        metrics = calculate_step_by_step_metrics(results, output_dir)
        
        logger.info("Standalone evaluation completed successfully!")
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("STANDALONE GTA BENCHMARK EVALUATION SUMMARY")
        print("="*70)
        print(f"Results File: {results_file}")
        print(f"Results saved to: {output_dir}")
        
        print("\nKey Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        print("✅ Evaluation completed! Check the output directory for detailed results.")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Standalone evaluation failed: {e}")
        raise


def check_answer_correctness_llm(pred_content: str, gt_content: Optional[dict | list], judge_llm: str='qwen3-8b') -> dict:
    """
    Use a LLM as a judge to determine if the prediction is correct.
    """

    output_format = JudgeResponse.model_json_schema()

    system_prompt = f"""
    You are a helpful assistant that have basic problem solving skills. You are
    tasked to determine whether the prediction correctly solves the problem by
    checking whether the prediction matches the ground truth answers. If the
    ground truth is a list, you should check whether the prediction matches
    one of the ground truth answers. If the ground truth is a dict, you should
    check whether the prediction matches *all* the answers in the whitelist, 
    while does not have *any* of the answers in the blacklist.
    """

    user_prompt = f"""
    Here is the prediction:
    {pred_content}

    Here is the ground truth:
    {gt_content}
    """
    config_file = 'llama_cpp_server_qwen8b.json'

    model = create_model_from_config(config_file)

    response = model.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        tools=[],
        output_format=output_format
    )
    del model
    return json.loads(response['message']['content'])['is_correct']


def main():
    """Main evaluation function."""
    args = parse_args()
    if args.standalone:
        if not args.results_file:
            logger.error("--results_file is required for standalone evaluation")
            sys.exit(1)
        if not args.output_dir:
            args.output_dir = f'outputs/standalone_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Run standalone evaluation
        run_standalone_evaluation(args.results_file, args.output_dir)
        return
    
    # Set up output directory using model_alias from config
    output_dir = setup_output_dir(args.config)
    
    logger.info("Starting Standalone GTA Benchmark Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset Path: {args.dataset_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Step-by-step mode: {args.step_by_step}")
    
    try:
        # Load dataset
        dataset = load_dataset(args.dataset_path, args.max_samples)
        
        # Create model
        model = create_model(args.config, args.model, args.temperature)
        
        # Run evaluation
        results = run_evaluation(
            model, dataset, output_dir, args.max_turns, args.save_steps, args.step_by_step, args.n_samples
        )
        
        # Calculate metrics
        if args.step_by_step:
            metrics = calculate_step_by_step_metrics(results, output_dir)
        else:
            # Extract data for traditional metrics calculation
            predictions = [r.get('prediction', []) for r in results]
            ground_truths = [r.get('ground_truth', []) for r in results]
            references = [r.get('reference') for r in results]
            metrics = calculate_metrics(predictions, ground_truths, references, output_dir)
        
        logger.info("Evaluation completed successfully!")
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("STANDALONE GTA BENCHMARK EVALUATION SUMMARY")
        print("="*70)
        print(f"Model: {args.model}")
        print(f"Configuration: {args.config}")
        print(f"Samples Evaluated: {len(dataset)}")
        print(f"Evaluation Mode: {'Step-by-step' if args.step_by_step else 'Standard'}")
        print(f"Results saved to: {output_dir}")
        
        print("\nKey Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        print("✅ Evaluation completed! Check the output directory for detailed results.")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Evaluation failed: Critical Error {e}")
        raise e
        sys.exit(1)





if __name__ == '__main__':
    main() 