import torch

class SoftwarePipeLine(object):
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stream = None
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        if self.stream is None:
            self.stream = torch.cuda.Stream()
            
        first = True
        for next_input, next_target, next_idx in self.dataloader:
            with torch.cuda.stream(self.stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_idx = next_idx.cuda(non_blocking=True)
            if not first:
                yield input, target, index
            else:
                first = False
            torch.cuda.current_stream().wait_stream(self.stream)
            input = next_input
            target = next_target
            index = next_idx
        yield input, target, index
