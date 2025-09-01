def evaluate_prompt(prompt, batch, adapter):
    """
    Given a prompt, batch of inputs, and an adapter, return predictions.
    batch: list of input strings (for SciERC, abstracts)
    """
    return [adapter.generate(prompt + " " + x) for x in batch]

def collect_errors(predictions, golds, batch):
    errors = []
    for inp, gold, pred in zip(batch, golds, predictions):
        # If inp is a tuple, take the first element (the input string)
        if isinstance(inp, tuple):
            inp_str = inp[0]
        else:
            inp_str = inp
        if pred != gold:
            errors.append((inp_str, gold, pred))
    return errors