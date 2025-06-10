"""
Standalone dataset implementation for GTA benchmark.
This version doesn't depend on OpenCompass framework.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import copy


def get_all_file_paths(directory: str) -> list:
    """Get all file paths in a directory recursively."""
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def organize_dialogs_correct(sample: dict, path: str) -> List[dict]:
    """
    Organize dialogs with proper tool call structure.
    This version maintains the correct OpenAI tool call format.
    """
    dialogs = []
    file_paths = get_all_file_paths(path)
    
    for item in sample['dialogs']:
        if item['role'] == 'tool':
            # Tool response - keep as is
            dialog = dict(
                role='tool',
                name=item['name'],
                content=item['content'],
            )
            dialogs.append(dialog)
        elif item['role'] == 'assistant' and 'tool_calls' in item.keys():
            # Assistant with tool calls - ensure proper format
            dialog = copy.deepcopy(item)
            
            # Update file paths to absolute paths if needed
            for tool_call in dialog.get('tool_calls', []):
                if 'function' in tool_call and 'arguments' in tool_call['function']:
                    for name, value in tool_call['function']['arguments'].items():
                        if isinstance(value, str) and os.path.join(path, value) in file_paths:
                            tool_call['function']['arguments'][name] = os.path.join(path, value)
            
            dialogs.append(dialog)
        else:
            # Regular user or assistant message
            dialogs.append(item)

    return dialogs


class GTADataset:
    """
    Standalone GTA Benchmark Dataset loader.
    This version doesn't depend on OpenCompass framework.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = self.load_data()
    
    def load_data(self) -> List[Dict]:
        """Load the GTA dataset."""
        data_file = self.data_path / 'dataset.json'
        if not data_file.exists():
            raise FileNotFoundError(f'Dataset file not found: {data_file}')

        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        
        data_list = []
        
        for idx, item in raw_data.items():
            idx = int(idx)
            
            # Extract tools in the correct format for OpenAI API
            tools = []
            for tool in item['tools']:
                tools.append(tool)  # Keep full tool metadata
            
            # Extract files
            files = [
                dict(type='file',
                     filetype=file['type'],
                     path=str((self.data_path / file['path']).absolute()))
                for file in item['files']
            ]
            
            gt_answer = item['gt_answer']
            
            # Organize dialogs with proper tool call structure
            organized_dialogs = organize_dialogs_correct(item, str(self.data_path.absolute()))
            
            sample = {
                'idx': idx,
                'dialogs': organized_dialogs,
                'tools': tools,
                'files': files,
                'gt_answer': gt_answer,
                'user_query': self._extract_user_query(organized_dialogs)
            }
            data_list.append(sample)
            
        return data_list
    
    def _extract_user_query(self, dialogs: List[Dict]) -> str:
        """Extract the first user query from dialogs."""
        for dialog in dialogs:
            if dialog['role'] == 'user':
                return dialog['content']
        return ""
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)


class GTAEvaluator:
    """
    Standalone evaluator for GTA benchmark that properly handles tool calls.
    This version doesn't depend on OpenCompass framework.
    """

    def __init__(self, mode='every_with_gt'):
        assert mode in ['every', 'every_with_gt']
        self.mode = mode
        
        # Try to import sentence transformer for similarity scoring
        try:
            from sentence_transformers import SentenceTransformer, util
            import numpy as np
            self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
            self.util = util
            self.np = np
            self.has_sentence_transformer = True
        except ImportError:
            print("Warning: sentence-transformers not available. Similarity scoring will be disabled.")
            self.has_sentence_transformer = False

    def bert_score(self, pred: str, gt: str) -> float:
        """Calculate BERT similarity score between prediction and ground truth."""
        if not self.has_sentence_transformer:
            return 0.0
            
        pred_emb = self.sentence_model.encode(pred, convert_to_tensor=True)
        gt_emb = self.sentence_model.encode(gt, convert_to_tensor=True)
        score = self.np.maximum(self.util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
        return score[0][0]

    @staticmethod
    def get_response_type(item):
        """Determine the type of response (tool call, answer, or tool return)."""
        if 'tool_calls' in item:
            return 'tool', item['tool_calls'][0]['function']
        elif item['role'] == 'assistant':
            return 'answer', item['content']
        else:
            return 'tool_return', item['content']

    @staticmethod
    def iscorrect(pred: str, ref: dict):
        """Check if prediction matches reference using whitelist/blacklist."""
        count = 0
        for aliases in ref['whitelist']:
            pattern = r'\b(?:' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
            if re.search(pattern, pred, re.IGNORECASE):
                count += 1
                
        if not ref['blacklist']:
            if count == len(ref['whitelist']):
                return True
        else:
            pattern_bk = r'\b(?:' + '|'.join(re.escape(alias) for aliases in ref['blacklist'] for alias in aliases) + r')\b'
            if count == len(ref['whitelist']) and not re.search(pattern_bk, pred, re.IGNORECASE):
                return True
        return False

    def simscore(self, pred: str, ref: list):
        """Calculate maximum similarity score against reference list."""
        if not self.has_sentence_transformer:
            return 0.0
            
        max_score = 0
        for s in ref:
            score = self.bert_score(pred, s)
            if score > max_score:
                max_score = score
        return max_score
    
    @staticmethod
    def gettype(name: str):
        """Get tool category type."""
        perception = ['OCR', 'ImageDescription', 'RegionAttributeDescription', 'TextToBbox']
        operation = ['DrawBox', 'AddText', 'GoogleSearch']
        logic = ['Calculator', 'Solver', 'Plot', 'MathOCR', 'CountGivenObject']
        creativity = ['TextToImage', 'ImageStylization']
        
        if name in perception:
            return 'perception'
        elif name in operation:
            return 'operation'
        elif name in logic:
            return 'logic'
        elif name in creativity:
            return 'creativity'
        else:
            return 'none'

    def evaluate(self, predictions: List[List[Dict]], ground_truths: List[List[Dict]], references: List[Any]) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of model predictions (conversation histories)
            ground_truths: List of ground truth conversation histories
            references: List of reference answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.mode == 'every_with_gt':
            total = {'tool': 0, 'answer': 0}
            metrics = {
                'inst_align': 0,
                'tool_acc': 0,
                'arg_acc': 0,
                'answer_acc': 0,
                'tool_call': 0,
                'tool_call_error': 0
            }
            
            for preds, gts, ref in zip(predictions, ground_truths, references):
                if ref:
                    total['answer'] += 1
                    
                # Compare each step in the conversation
                min_len = min(len(preds), len(gts))
                for i in range(min_len):
                    pred = preds[i]
                    gt = gts[i]
                    
                    pred_type, pred_content = self.get_response_type(pred)
                    gt_type, gt_content = self.get_response_type(gt)
                    
                    # Check instruction alignment
                    if pred_type == gt_type and 'error' not in pred:
                        metrics['inst_align'] += 1
                    
                    # Count tool calls
                    if gt_type == 'tool':
                        total['tool'] += 1
                    if pred_type == 'tool':
                        metrics['tool_call'] += 1
                        if 'error' in pred:
                            metrics['tool_call_error'] += 1
                    
                    # Evaluate tool accuracy
                    if pred_type == gt_type == 'tool' and pred_content['name'] == gt_content['name']:
                        metrics['tool_acc'] += 1
                        if pred_content['arguments'] == gt_content['arguments']:
                            metrics['arg_acc'] += 1
                    
                    # Evaluate answer accuracy
                    elif pred_type == gt_type == 'answer':
                        if isinstance(ref, dict):
                            metrics['answer_acc'] += self.iscorrect(pred_content, ref)
                        elif isinstance(ref, list):
                            metrics['answer_acc'] += self.simscore(pred_content, ref)
                            
            # Calculate final scores
            return {
                'inst_align': metrics['inst_align'] / sum(total.values()) * 100 if sum(total.values()) > 0 else 0,
                'tool_acc': metrics['tool_acc'] / total['tool'] * 100 if total['tool'] > 0 else 0,
                'arg_acc': metrics['arg_acc'] / total['tool'] * 100 if total['tool'] > 0 else 0,
                'answer_acc': metrics['answer_acc'] / total['answer'] * 100 if total['answer'] > 0 else 0,
                'tool_call': metrics['tool_call'],
                'tool_call_error': metrics['tool_call_error'],
                'total_samples': len(predictions),
                'total_tool_calls': total['tool'],
                'total_answers': total['answer']
            }
            
        else:
            # Simplified evaluation mode
            return {
                'precision_perception': 0,
                'recall_perception': 0,
                'f1_perception': 0,
                'precision_operation': 0,
                'recall_operation': 0,
                'f1_operation': 0,
                'precision_logic': 0,
                'recall_logic': 0,
                'f1_logic': 0,
                'precision_creativity': 0,
                'recall_creativity': 0,
                'f1_creativity': 0,
                'total_samples': len(predictions)
            } 