import unittest
from unittest.mock import MagicMock, patch
import warnings

from colab_persist import accounts


QUOTA_WARNING = (
    "Your application has authenticated using end user credentials from Google "
    "Cloud SDK without a quota project. You might receive a 'quota exceeded' "
    "or 'API not enabled' error."
)


class CredentialWarningTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"expected_email": "test@example.com",
                    "credentials_file": "/unused/private-adc.json"}
        self.credentials = object()

    def verify(self, load_warnings=(), response_warnings=()):
        def load(*_args, **_kwargs):
            for message, category in load_warnings:
                warnings.warn(message, category)
            return self.credentials, None

        def get(*_args, **_kwargs):
            for message, category in response_warnings:
                warnings.warn(message, category)
            response = MagicMock()
            response.json.return_value = {"email": self.cfg["expected_email"]}
            return response

        with patch.object(accounts.google.auth, "load_credentials_from_file", side_effect=load) as loader, \
             patch.object(accounts, "AuthorizedSession") as session, \
             patch.object(accounts, "_get_adc_credentials", side_effect=AssertionError("Unexpected global ADC")), \
             warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            session.return_value.__enter__.return_value.get.side_effect = get
            self.assertIs(accounts.verified_credentials(self.cfg), self.credentials)
            loader.assert_called_once_with(self.cfg["credentials_file"], scopes=list(accounts.SCOPES))
            session.assert_called_once_with(self.credentials)
        return observed

    def test_user_adc_quota_warning_is_suppressed_during_private_load(self):
        self.assertEqual(self.verify([(QUOTA_WARNING, UserWarning)]), [])

    def test_unrelated_user_warning_is_preserved(self):
        observed = self.verify([(QUOTA_WARNING, UserWarning), ("Credential diagnostic", UserWarning)])
        self.assertEqual([str(item.message) for item in observed], ["Credential diagnostic"])

    def test_same_message_with_other_warning_category_is_preserved(self):
        observed = self.verify([(QUOTA_WARNING, RuntimeWarning)])
        self.assertEqual([str(item.message) for item in observed], [QUOTA_WARNING])
        self.assertIs(observed[0].category, RuntimeWarning)

    def test_filter_does_not_cover_identity_request(self):
        observed = self.verify(response_warnings=[(QUOTA_WARNING, UserWarning)])
        self.assertEqual([str(item.message) for item in observed], [QUOTA_WARNING])

    def test_filter_is_restored_when_loading_fails(self):
        def fail(*_args, **_kwargs):
            warnings.warn(QUOTA_WARNING, UserWarning)
            raise ValueError("Invalid test credential")

        with patch.object(accounts.google.auth, "load_credentials_from_file", side_effect=fail), \
             patch.object(accounts, "AuthorizedSession") as session, \
             warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(RuntimeError, "Could not verify Google credentials"):
                accounts.verified_credentials(self.cfg)
            warnings.warn(QUOTA_WARNING, UserWarning)
            session.assert_not_called()
        self.assertEqual([str(item.message) for item in observed], [QUOTA_WARNING])


if __name__ == "__main__":
    unittest.main()
