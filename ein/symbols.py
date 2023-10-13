_index_ids: dict["Index", int] = {}
_variable_ids: dict["Variable", int] = {}


class Index:
    def __repr__(self) -> str:
        if self not in _index_ids:
            _index_ids[self] = len(_index_ids)
        return f"@{_index_ids[self]}"


class Variable:
    def __repr__(self) -> str:
        if self not in _variable_ids:
            _variable_ids[self] = len(_variable_ids)
        return f"${_variable_ids[self]}"
