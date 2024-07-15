from peft import PeftModel

class PeftMixin:
    @property
    def peft_model(self) -> 'PeftModel':
        return self._peft_model[0]

    @property
    def peft_model_prefix(self):
        return self._peft_model_prefix

    # @peft_model.setter
    # def peft_model(self, value):
    #     self._peft_model = value

    def set_peft_model(self, value: 'PeftModel', prefix: str = ''):
        """
        Args:
            prefix: must end with "." if nonempty
        Returns:
        """
        self._peft_model = (value, )
        prefix_remove_dot = prefix[:-1] if prefix.endswith('.') else prefix
        assert self.get_submodule(prefix_remove_dot) is value.base_model.model
        prefix_with_dot = prefix if prefix == '' or prefix.endswith('.') else f'{prefix}.'
        self._peft_model_prefix = prefix_with_dot

    def __setattr__(self, name: str, value: ...) -> None:
        if name == 'peft_model':
            # let nn.Module not register it as a submodule
            value = (value, )
        super().__setattr__(name, value)
