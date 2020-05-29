from flask import jsonify, make_response


def wrap_response(data, errors=None, code=200):
    body = data
    body['isSuccessStatusCode'] = not bool(errors)
    body['statusCode'] = code
    if not errors:
        return make_response(jsonify(body), code)
    else:
        return make_response(jsonify(body), code)
