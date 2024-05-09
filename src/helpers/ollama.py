import subprocess
import shutil


class OllamaHelper:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def is_ollama_available(self) -> bool:
        """Check if ollama is installed."""
        if shutil.which("ollama") is None:
            # raise FileNotFoundError("ollama is not installed. Please install it first.")
            return False
        return True

    def pull_ollama_model(self, ollama_model: str) -> None:
        """Pull the ollama model."""
        if not self.is_ollama_available():
            raise FileNotFoundError("ollama is not installed. Please install it first.")

        command = f"ollama pull {ollama_model}"

        # Execute the command and wait for it to finish
        try:
            if self.verbose:
                print(f"Pulling the model: {ollama_model}...")
            result = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )

            if (  # Check if the output contains "success"
                "success" not in result.stdout.lower()
                and "success" not in result.stderr.lower()
            ):
                raise ValueError(
                    f"Error executing command: '{command}'\n"
                    f"Command failed with output: {result.stderr}"
                )

        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Error executing command: {e}\n"
                "Make sure the model exists and ollama is up to date."
            )

        if self.verbose:
            print(f"Successfully pulled the model: {ollama_model}")
