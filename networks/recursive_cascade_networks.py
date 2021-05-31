from .base_networks import *
from .spatial_transformer import SpatialTransform


class RecursiveCascadeNetwork(nn.Module):
    def __init__(self, n_cascades, im_size=(512, 512)):
        super(RecursiveCascadeNetwork, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stems = []
        self.stems.append(VTNAffineStem())
        for i in range(n_cascades):
            self.stems.append(VTN(flow_multiplier=1.0 / n_cascades))

        # Parallelize across all available GPUs
        if torch.cuda.device_count() > 1:
            self.stems = [nn.DataParallel(model) for model in self.stems]

        for model in self.stems:
            model.to(device)

        self.reconstruction = SpatialTransform(im_size)
        self.reconstruction = nn.DataParallel(self.reconstruction)
        self.reconstruction.to(device)

    def forward(self, fixed, moving):
        flows = []
        stem_results = []
        # Affine registration
        flow, W, b, det_loss = self.stems[0](fixed, moving)
        stem_results.append(self.reconstruction(moving, flow))
        flows.append(flow)
        for model in self.stems[1:]: # cascades
            # registration between the fixed and the warped from last cascade
            flow = model(fixed, stem_results[-1])
            stem_results.append(self.reconstruction(stem_results[-1], flow))
            flows.append(flow)

        return stem_results, flows

