from collections import Counter

def evaluate(selected,loads):

    print("\n===== METRICS =====")

    unique_branches = len(set(load["branch"] for load in selected))
    print(f"Unique branches: {unique_branches}")

    print("\nBranch distribution:")
    branch_counts = Counter(load["branch"] for load in selected)

    for branch, count in sorted(branch_counts.items()):
        print(f"Branch {branch:>2}: {count}")

    backbone = sum(load["on_backbone"] for load in selected)
    print(
        f"\nBackbone meters: "
        f"{backbone}/{len(selected)} "
        f"({100 * backbone / len(selected):.1f}%)"
    )
    print("\nBackbone buses selected:")
    for load in selected:
        if load["on_backbone"]:
            print(load["bus"], load["base_kw"], load["score"])
    selected_backbone = {
        load["bus"]
        for load in selected
        if load["on_backbone"]
    }

    eligible_backbone = {
        load["bus"]
        for load in loads
        if load["on_backbone"]
    }

    print(f"Eligible backbone buses : {len(eligible_backbone)}")
    print(f"Selected backbone buses : {len(selected_backbone)}")
    print(
        f"Backbone coverage        : "
        f"{100 * len(selected_backbone) / len(eligible_backbone):.1f}%"
    )

    avg_depth = sum(load["depth"] for load in selected) / len(selected)
    print(f"Average depth: {avg_depth:.2f}")
    print(max(load["depth"] for load in loads))

    total_downstream = sum(load["downstream_kw"] for load in selected)
    avg_downstream = total_downstream / len(selected)

    print(f"Total downstream kW: {total_downstream:.1f}")
    print(f"Average downstream kW: {avg_downstream:.1f}")


    print("\nSize classes:")
    size_counts = Counter(load["size_class"] for load in selected)

    for size, count in sorted(size_counts.items()):
        print(f"{size:<12}: {count}")

