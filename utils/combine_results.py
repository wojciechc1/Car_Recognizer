#S=wc×pc+wl×pl

def combine_results(results):
    WEIGHT_CONTEXT = 0.6
    WEIGHT_LOGO = 0.4

    scores = {}

    for view_name, view_data in results.items():
        # Context detection
        if "context" in view_data:
            ctx_label = view_data["context"]["label_name"]
            ctx_conf = view_data["context"]["probabilities"][view_data["context"]["label_idx"]]
            scores[ctx_label] = scores.get(ctx_label, 0) + ctx_conf * WEIGHT_CONTEXT

        # Logo detection
        for logo in view_data.get("logo", []):
            logo_label = logo["label"]
            logo_conf = logo["confidence"]
            scores[logo_label] = scores.get(logo_label, 0) + logo_conf * WEIGHT_LOGO

    # Marka z najwyższym wynikiem
    final_brand = max(scores, key=scores.get)
    print("Final brand:", final_brand)
    print("Scores:", scores)

    return {"final_brand": final_brand,
            "scores": scores}

