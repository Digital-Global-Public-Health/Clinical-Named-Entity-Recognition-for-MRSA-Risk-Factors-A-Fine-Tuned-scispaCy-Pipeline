# src/utils_db.py
"""SAP HANA connection utilities for the MRSA NLP NER-based pipeline."""

import os
import logging
from dotenv import load_dotenv

LOG = logging.getLogger("mrsa_nlp.ner.db")


def airms_conn_kwargs() -> dict:
    """
    Load HANA connection parameters from environment variables (or a .env file).

    Expected environment variables (see .env.example):
      AIRMS_HOST, AIRMS_PORT, AIRMS_USER, AIRMS_PASSWORD,
      AIRMS_DATABASE, AIRMS_ENCRYPT, AIRMS_SSL_VALIDATE_CERTIFICATE,
      AIRMS_SSL_HOSTNAME_IN_CERT, AIRMS_SSL_TRUSTSTORE, AIRMS_CONNECT_TIMEOUT

    Returns
    -------
    dict
        Keyword arguments ready to be passed to
        ``hana_ml.dataframe.ConnectionContext(**kwargs)``.
    """
    load_dotenv()
    return {
        "address":                  os.getenv("AIRMS_HOST", "127.0.0.1"),
        "port":                     int(os.getenv("AIRMS_PORT", "54321")),
        "user":                     os.getenv("AIRMS_USER"),
        "password":                 os.getenv("AIRMS_PASSWORD"),
        "databaseName":             os.getenv("AIRMS_DATABASE", "AIRMS"),
        "encrypt":                  os.getenv("AIRMS_ENCRYPT", "TRUE"),
        "sslValidateCertificate":   os.getenv("AIRMS_SSL_VALIDATE_CERTIFICATE", "FALSE"),
        "sslHostNameInCertificate": os.getenv("AIRMS_SSL_HOSTNAME_IN_CERT", "hana-pa2.mssm.edu"),
        "sslTrustStore":            os.getenv("AIRMS_SSL_TRUSTSTORE", "None"),
        "connectTimeout":           int(os.getenv("AIRMS_CONNECT_TIMEOUT", "0")),
    }


def connect_hana():
    """
    Create and return a live SAP HANA ConnectionContext.

    Uses ``airms_conn_kwargs()`` to source credentials. The SSH tunnel
    must be established before calling this function (see
    scripts/start_airms_tunnel.sh or scripts/run_cohort_builder.sh).

    Returns
    -------
    hana_ml.dataframe.ConnectionContext
        An open HANA connection context.

    Raises
    ------
    ImportError
        If the ``hana-ml`` package is not installed.
    hana_ml.dataframe.ConnectionError
        If the connection cannot be established (wrong credentials, tunnel
        not running, etc.).
    """
    from hana_ml.dataframe import ConnectionContext
    kwargs = airms_conn_kwargs()
    LOG.info(
        "Connecting to HANA at %s:%s (DB=%s, encrypt=%s, sslValidateCert=%s)",
        kwargs["address"], kwargs["port"], kwargs["databaseName"],
        kwargs["encrypt"], kwargs["sslValidateCertificate"],
    )
    return ConnectionContext(**kwargs)
