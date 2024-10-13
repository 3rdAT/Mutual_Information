import torch

def process_logits(logits_tensor):
    """
    Process logits in tensor form, allowing operations across the entire 
    sequence. The tensor should have shape (num_steps, batch_size, vocab_size).
    """
    # Example: Convert logits to probabilities for the entire sequence
    probabilities_tensor = torch.softmax(logits_tensor, dim=-1)

    # Example: Convert logits to log probabilities for the entire sequence
    log_probs_tensor = torch.log_softmax(logits_tensor, dim=-1)

    return probabilities_tensor, log_probs_tensor

