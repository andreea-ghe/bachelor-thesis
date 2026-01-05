import sys


class DuplicateStdoutFileManager(object):
    def __init__(self, filepath):
        """
        Where to duplicate stdout output to a file.
        """
        self.filepath = filepath
        self.mode = 'w+'

    def __enter__(self):
        """
        Called when entering the with statement.
        Redirects stdout to a file writer that duplicates output.
        """
        self._stdout = sys.stdout # save original stdout
        self._file = DuplicateStdoutFileWriter(self.filepath, self.mode, self._stdout)
        sys.stdout = self._file # redirect stdout to file writer

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Called when exiting the with statement.
        Restores original stdout.
        """
        sys.stdout = self._stdout
        self._file.flush()
        self._file._file.close()


class DuplicateStdoutFileWriter(object):
    def __init__(self, filepath, mode, stdout):
        """

        """
        self.filepath = filepath
        self._stdout = stdout
        self._content = ""
        self.encoding = None
        self._file = open(self.filepath, mode)

    def write(self, message):
        """
        The method called when printing happens.
        """
        while "\n" in message:
            # splitting on new lines ensures flush on every line
            # -> line-by-line logging
            # -> terminal output is not delayed
            pos = message.find("\n")
            self._content += message[:pos + 1]
            self.flush()
            message = message[pos + 1 :]

        self._content += message
        if len(self._content) > 1000:
            self.flush()

    def flush(self):
        """
        Flush the content buffer to both terminal and file.
        """
        self._stdout.write(self._content) # write buffer to terminal
        self._stdout.flush()

        self._file.write(self._content) # write buffer to file
        self._file.flush()
        
        self._content = ""

    def __del__(self):
        self._file.close()