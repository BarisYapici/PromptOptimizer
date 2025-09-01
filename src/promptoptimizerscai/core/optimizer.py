class PromptOptimizer:
    def __init__(self, model_adapter):
        self.adapter = model_adapter

    def optimize_step(self, prompt, batch):
        """
        One optimization step: evaluate, collect errors, get feedback, edit prompt.
        batch: list of (input, gold_label)
        """
        inputs = [x[0] for x in batch]
        golds = [x[1] for x in batch]
        # 1. Run prompt on batch
        preds = [self.adapter.generate(prompt + " " + inp) for inp in inputs]
        # 2. Collect errors
        errors = []
        for inp, gold, pred in zip(inputs, golds, preds):
            if pred != gold:
                errors.append((inp, gold, pred))
        # 3. Get gradient feedback from adapter
        if hasattr(self.adapter, "generate_gradient"):
            feedback = self.adapter.generate_gradient(prompt, errors)
        else:
            feedback = ["Improve the prompt."]
        # 4. Edit prompt using feedback
        if hasattr(self.adapter, "edit_prompt"):
            new_prompts = self.adapter.edit_prompt(prompt, feedback[0])
        else:
            # Fallback: just append feedback
            new_prompts = [prompt + " " + feedback[0]]
        # 5. Return first new prompt
        return new_prompts[0]