Global-Basis Mixture of Experts

Compress Mixture-of-Experts (MoE) FFNs by replacing per-expert "up/gate" weights with mixtures of a single, global bank of bases, while keeping inference fast via precomposition + caching. Enable sparse (top-s) mixtures so the global bank can be large without making runtime heavy.

I think the idea holds water, but I couldn't really get the alternating minimization optimization loop working well. Might come back to it someday.
