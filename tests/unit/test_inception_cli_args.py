import pytest

from kestrel_sovereign import inception_service


def test_inception_cli_accepts_output_alias():
    parser = inception_service.build_cli_parser()
    args = parser.parse_args(["--output", "/tmp", "--test", "--name", "X"])
    assert str(args.output_dir) == "/tmp"


def test_inception_cli_disallows_abbrev_prefixes():
    parser = inception_service.build_cli_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--out", "/tmp"])
