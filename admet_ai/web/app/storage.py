"""Contains global variables for use in the ADMET-AI website."""

import time

import pandas as pd

from admet_ai.web.app import app

USER_TO_PREDS: dict[str, pd.DataFrame] = {}
USER_TO_LAST_ACTIVITY: dict[str, float] = {}


def get_user_preds(user_id: str) -> pd.DataFrame:
    """Gets the user's predictions.

    :param user_id: The user's session ID.
    :return: A DataFrame of the user's predictions, or an empty DataFrame if none.
    """
    return USER_TO_PREDS.get(user_id, pd.DataFrame())


def set_user_preds(user_id: str, preds_df: pd.DataFrame) -> None:
    """Sets the user's predictions.

    :param user_id: The user's session ID.
    :param preds_df: A DataFrame of predictions to store for the user.
    """
    USER_TO_PREDS[user_id] = preds_df


def update_user_activity(user_id: str) -> None:
    """Updates the user's last activity time.

    :param user_id: The user's session ID.
    """
    USER_TO_LAST_ACTIVITY[user_id] = time.time()


def cleanup_storage() -> None:
    """Clean up storage by removing data from users that are no longer active.

    Runs in an infinite loop; removes predictions for users inactive longer than the session lifetime.
    """
    print("Starting cleanup")

    while True:
        # Wait for cleanup_frequency minutes
        cleanup_frequency = app.config["SESSION_LIFETIME"]
        time.sleep(cleanup_frequency)

        # Remove data from users that have been inactive
        num_removed = 0
        now = time.time()
        for user_id, last_activity in list(USER_TO_LAST_ACTIVITY.items()):
            if now - last_activity > cleanup_frequency:
                del USER_TO_PREDS[user_id]
                del USER_TO_LAST_ACTIVITY[user_id]
                num_removed += 1

        print(f"Cleanup removed data from {num_removed:,} users with {len(USER_TO_PREDS):,} users remaining.")
