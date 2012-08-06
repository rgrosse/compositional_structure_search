
tokens = ('LETTER', 'PLUS', 'LPAREN', 'RPAREN', 'GSM')

t_LETTER = r'[gmbcMBC]'
t_PLUS = r'\+'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_GSM = r's'

t_ignore = ' '

def t_error(t):
    raise RuntimeError("Illegal character: '%s'" % t.value[0])

import ply.lex as lex
lex.lex()

def p_expression_plus(t):
    """expression : expression PLUS term"""
    t[0] = ('+', t[1], t[3])

def p_expression_term(t):
    """expression : term"""
    t[0] = t[1]

def p_term_times(t):
    """term : factor factor"""
    t[0] = ('*', t[1], t[2])

def p_term_factor(t):
    """term : factor"""
    t[0] = t[1]

def p_factor_gsm(t):
    """factor : GSM LPAREN expression RPAREN"""
    t[0] = ('s', t[3])

def p_factor_group(t):
    """factor : LPAREN expression RPAREN"""
    t[0] = t[2]

def p_factor_letter(t):
    """factor : LETTER"""
    t[0] = t[1]

def p_error(t):
    raise RuntimeError("Syntax error at '%s'" % t[1])

import ply.yacc as yacc
yacc.yacc()

def parse(s):
    return yacc.parse(s)


