import subprocess


def test_tdr_help():
    """Basic smoke test: tdr --help should run without error."""
    result = subprocess.run(
        ["tdr", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Help output should mention the process subcommand
    assert "process" in result.stdout
