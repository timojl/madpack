from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy

bce_logits = binary_cross_entropy_with_logits
ce_logits = binary_cross_entropy_with_logits

__all__ = ['binary_cross_entropy_with_logits', 'binary_cross_entropy', 'cross_entropy', 'bce_logits', 'ce_logits']
