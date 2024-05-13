import yaml


class Config:
    def __init__(self, file_path: str):
        """Initialize the Config class with the configuration data.
        Args:
            file_path (str): The path to the configuration file.
        """
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        self.llm = config_data.get("llm", {})
        self.embedding = config_data.get("embedding", {})
        self.vectorstore = config_data.get("vectorstore", {})
        self.readers = config_data.get("readers", {})
