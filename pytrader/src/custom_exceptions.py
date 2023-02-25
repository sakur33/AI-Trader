class InvalidCredentials(ValueError):
    pass


class LoginError(ValueError):
    pass


class SocketError(ValueError):
    pass


class ApiException(ValueError):
    pass


class DbError(ValueError):
    pass


class LogicError(ValueError):
    pass


class SessionError(ValueError):
    pass
