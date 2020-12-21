import torch


class YNet(torch.nn.Module):
    """Module that calls forward functions of child modules in parallel.

    When the `forward` method of this module is called, all the
    arguments are forwarded to the 'forward' method of the joint part.
    Then the output is fed to the two separate parts

    The returned values from the child modules are returned as a tuple.

    Args:
        *modules: Child modules. Each module should be callable.
    """

    def __init__(self, *modules):
        """
        The first item in modules should be the joint part. The rest are the individual heads.
        :param modules:
        """
        super().__init__()
        self.child_modules = torch.nn.ModuleList(modules)


    def forward(self, *args, **kwargs):
        """Forward the arguments to the child modules.

        Args:
            *args, **kwargs: Any arguments forwarded to child modules.  Each
                child module should be able to accept the arguments.

        Returns:
            tuple: Tuple of the returned values from the child modules.
        """
        joint_module, heads = self.child_modules[0], self.child_modules[1:]
        joint_in = joint_module(*args, **kwargs)
        return tuple(mod(joint_in) for mod in heads)
