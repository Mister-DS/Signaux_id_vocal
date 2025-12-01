import argparse
import sys
from enroll import EnrollmentManager
from validate import ValidationManager
from live_auth import LiveAuthenticator

GMM_THRESHOLD = 5.0
DTW_THRESHOLD = 0.25


def main():
    parser = argparse.ArgumentParser(
        description="Système d'Authentification Vocale")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_enroll = subparsers.add_parser(
        "enroll", help="Entrainer tous les modèles")

    parser_validate = subparsers.add_parser(
        "validate", help="Lancer la validation")

    parser_auth = subparsers.add_parser(
        "auth", help="Enregistrer et authentifier un sample en direct")
    parser_auth.add_argument("--target", type=str, default=None,
                             help="Forcer la comparaison avec l'utilisateur spécifié")

    args = parser.parse_args()

    if args.command == "enroll":
        manager = EnrollmentManager()
        manager.run_enrollment()

    elif args.command == "validate":
        manager = ValidationManager(
            gmm_threshold=GMM_THRESHOLD, dtw_threshold=DTW_THRESHOLD)
        manager.run_benchmark()

    elif args.command == "auth":
        authenticator = LiveAuthenticator(
            gmm_threshold=GMM_THRESHOLD,
            dtw_threshold=DTW_THRESHOLD
        )
        authenticator.run(target_user=args.target)


if __name__ == "__main__":
    main()
