'''


1. ViewClassifier -> 'front', 'side', 'rear', 'angled'

- (front) - ContextClassifier - LogoDetector
- (side) - CarTypeClassifier
- (back) - [ContextClassifier] - LogoDetector - ModelDetector
- (angled) - CarTypeClassifier - LogoDetector

'''