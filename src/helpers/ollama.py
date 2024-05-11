import time
import shutil
import subprocess

import ollama

from src.helpers import web_healthcheck


class OllamaHelper:
    def __init__(self, base_url: str = "http://localhost:11434", verbose: bool = True):
        """Initialize the OllamaHelper class.

        Args:
            base_url (str, optional): The base URL of the Ollama server. Defaults to "http://localhost:11434".
            verbose (bool, optional): Whether to print the output of the commands. Defaults to True.

        Raises:
            FileNotFoundError: If ollama is not installed.
        """
        self.base_url = base_url
        self.verbose = verbose

        if not self._is_installed():
            raise FileNotFoundError("ollama is not installed. Please install it first.")

        self._wake_up()

    def _is_installed(self) -> bool:
        """Check if ollama is installed.

        Returns:
            bool: True if ollama is installed, False otherwise.
        """
        if shutil.which("ollama") is None:
            # raise FileNotFoundError("ollama is not installed. Please install it first.")
            return False
        return True

    def _wake_up(self) -> None:
        """Start the Ollama server if it is not running.
        If ollama is not running try to start it with "ollama serve"
        and check again if it is running
        """
        ollama_running = web_healthcheck(self.base_url)
        if not ollama_running:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            retries = 5
            while not ollama_running and retries > 0:
                retries -= 1
                ollama_running = web_healthcheck(self.base_url)
                if not ollama_running:
                    time.sleep(2)  # wait for server to start

            if not ollama_running:
                raise Exception("Unable to start Ollama server")

    @property
    def available_models(self) -> list[str]:
        """List all the available ollama models.

        Returns:
            list[str]: List of available ollama models.
        """
        try:
            ollama_artifacts = ollama.list()
            if "models" in ollama_artifacts:
                models = ollama_artifacts["models"]
                return [model["name"] for model in models]
            return []
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error listing models: {e}")

    def model_exists(self, ollama_model: str) -> bool:
        """Check if the ollama model exists.

        Args:
            ollama_model (str): The model to check.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        # If the ollama_model does not include ":" add ":latest"
        if ":" not in ollama_model:
            ollama_model += ":latest"

        return ollama_model in self.available_models

    def remove_model(self, ollama_model: str) -> None:
        """Remove the ollama model.

        Args:
            ollama_model (str): The model to remove.

        Returns:
            None
        """
        if not self._is_installed():
            raise FileNotFoundError("ollama is not installed. Please install it first.")

        # check if the model is in the available models
        if not self.model_exists(ollama_model):
            print(f"Model {ollama_model} does not exist.")
            return

        command = f"ollama rm {ollama_model}"

        # Execute the command and wait for it to finish
        try:
            if self.verbose:
                print(f"Removing the model: {ollama_model}...")
            result = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )

            if (  # Check if the output contains "success"
                "deleted" not in result.stdout.lower()
                and "deleted" not in result.stderr.lower()
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
            print(f"Successfully removed the model: {ollama_model}")

    def pull_model(self, ollama_model: str, force: bool = False) -> None:
        """Pull the ollama model.

        Args:
            ollama_model (str): The model to pull.
            force (bool, optional): Whether to force pull the model. Defaults to False.

        Returns:
            None
        """
        # if force is True, remove the model first
        if force:
            self.remove_model(ollama_model)
        else:  # Check if the model is already available
            if self.model_exists(ollama_model):
                if self.verbose:
                    print(f"Model {ollama_model} is already available.")
                return

        # Try to pull the model
        try:
            if self.verbose:
                print(f"Pulling the model: {ollama_model}...")

            response = ollama.pull(ollama_model)
            if response["status"] != "success":
                raise ValueError(
                    f"Error pulling the model: {ollama_model}. Response: {response}"
                )

        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Error executing command: {e}\n"
                "Make sure the model exists and ollama is up to date."
            )

        if self.verbose:
            print(f"Successfully pulled the model: {ollama_model}")
