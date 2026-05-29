import numpy as np

class PlacementScorer:
    def __init__(self, overlap_penalty: float = 2.0,
                 coverage_reward: float = 1.0,
                 gap_penalty: float = 0.5,
                 weight_multiplier: float = 1.0):
        # weight_multiplier skaliert die Per-Pixel-Gewichte (typisch 1/scale^2),
        # sodass der score_max unabhaengig von der Aufloesung bleibt.
        self.overlap_penalty = overlap_penalty * weight_multiplier
        self.coverage_reward = coverage_reward * weight_multiplier
        self.gap_penalty = gap_penalty * weight_multiplier
    
    def score(self, rendered: np.ndarray, target: np.ndarray) -> float:
        """
        Score a rendered guess against target.
        
        Returns:
            Score (higher is better)
        """
        # Overlap: pixels where multiple pieces are (value > 1)
        overlap = np.sum(rendered > 1)
        
        # Coverage: pixels that match target exactly
        coverage = np.sum((rendered == 1) & (target > 0))
        
        # Gaps: target pixels not covered
        gaps = np.sum((rendered == 0) & (target > 0))
        
        # Calculate score
        score = (self.coverage_reward * coverage 
                 - self.overlap_penalty * overlap 
                 - self.gap_penalty * gaps)
        
        return score