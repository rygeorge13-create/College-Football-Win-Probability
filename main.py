import argparse
from src.cfb_model import CFBModel


def main():
    parser = argparse.ArgumentParser(
        description="College football win probability model"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("update")
    sub.add_parser("top25")
    sub.add_parser("evaluate")
    sub.add_parser("calibration")
    info_parser = sub.add_parser("info")
    info_parser.add_argument("team")

    check_parser = sub.add_parser("check")
    check_parser.add_argument("team")

    team_parser = sub.add_parser("team")
    team_parser.add_argument("team")

    game_parser = sub.add_parser("game")
    game_parser.add_argument("team_a")
    game_parser.add_argument("team_b")

    sim_parser = sub.add_parser("simulate")
    sim_parser.add_argument("team")
    sim_parser.add_argument("--n", type=int, default=5000)

    args = parser.parse_args()

    model = CFBModel()

    # Rebuild the model using the current data.
    model.update()

    if args.command == "update":
        print("Model updated.")
        print(model.last_update_summary)

    elif args.command == "top25":
        print(model.top25().to_string(index=False))

    elif args.command == "calibration":
        print(
            model.calibration().to_string(index=False)
        )

    elif args.command == "evaluate":
        print(model.evaluate().to_string(index=False))

    elif args.command == "check":
        print(model.check_team(args.team).to_string(index=False))

    elif args.command == "team":
        print(model.team(args.team).to_string(index=False))

    elif args.command == "info":
        print(model.info(args.team).to_string(index=False))

    elif args.command == "game":
        print(
            model.game(
                args.team_a,
                args.team_b
            ).to_string(index=False)
        )

    elif args.command == "simulate":
        print(
            model.simulate_team(
                args.team,
                args.n
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()