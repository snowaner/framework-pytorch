#! /usr/bin/env python
#################################################################################
#     File Name           :     config.py
#     Created By          :     WanCQ
#     Creation Date       :     [2018-12-02 04:10]
#     Last Modified       :     [2018-12-02 04:16]
#     Description         :     config file related
#################################################################################

###################### Config file Transform Class###############################
# this class will transform the config file into a more functional result. It
# supports the options in form of list and dictionary. It defines to fetch
# params in config file, and sections/options/has_params function to search
# whether the params exist.
class ConfigFile(object):
    def __init__(self, config):
        self.params = {}
        self._Params_Type_Transform(config)

    def _Params_Type_Transform(self, config):
        sections = config.sections()
        for sec in sections:
            options = config.options(sec)
            self.params[sec] = {}
            for opt in options:
                self.params[sec][opt] = self._Type_Transform(config.get(sec, opt))

    def _Type_Transform(self, param):
        if not isinstance(param, str):
            raise ValueError('Config Params are not in type of string, some \
                             bugs may occur in codes')
        # type str
        if param[0] == '\'' and param[-1] == '\'':
            return param[1:-1]
        # type list
        elif param[0] == '[' and param[-1] == ']':
            lst = []
            ll = param[1:-1].split(',')
            for l in ll:
                lst.append(self._Type_Transform(l.replace(' ', '')))
            return lst
        # type dictionary
        elif param[0] == '{' and param[-1] == '}':
            raise NotImplementedError
        # type tuple
        elif param[0] == '(' and param[-1] == ')':
            raise NotImplementedError
        elif param == 'true' or param == 'True' or param == 'TRUE':
            return True
        elif param == 'false' or param == 'False' or param == 'FALSE':
            return False
        # type int
        else:
            if param.count('.') > 1:
                raise ValueError('Illegal params in config file, please \
                                    check the type of configures')
            for p in param:
                if p != '.' and (p < '0' or p > '9'):
                    raise ValueError('Illegal params in config file, please \
                                     check the type of configures')
            if param.count('.') == 0:
                return int(param)
            return float(param)

    def get(self, sec, opt):
        if self.has_params(sec, opt):
            return self.params[sec][opt]
        return None

    def sections(self):
        return self.params.keys()

    def options(self, sec):
        return self.params[sec].keys()

    def __getitem__(self, sec, param):
        return self.params[sec][opt]

    def has_params(self, sec, opt):
        if self.params.has_key(sec):
            if self.params[sec].has_key(opt):
                return True
        return False
