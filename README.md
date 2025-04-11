#include <iostream>
#include <cctype>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <memory>

#include <fstream>
#include <sstream>

using namespace std;

/*******************************************
 * 1. DefiniciÃ³n de tokens (AnÃ¡lisis LÃ©xico)
 *******************************************/

// Tipos de token
enum TokenType {
    // Palabras clave
    T_INT,      // palabra reservada "int"
    T_FLOAT,    // palabra reservada "float"
    T_CHAR,     // palabra reservada "char"
    T_VOID,     // palabra reservada "void"
    T_IF,       // palabra reservada "if"
    T_ELSE,     // palabra reservada "else"
    T_WHILE,    // palabra reservada "while"
    T_FOR,      // palabra reservada "for"
    T_RETURN,   // palabra reservada "return"
    T_COUT,     // palabra reservada "cout"
    T_CIN,      // palabra reservada "cin"
    T_INCLUDE,  // palabra reservada "include"
    T_USING,    // palabra reservada "using"
    T_NAMESPACE,// palabra reservada "namespace"
    T_STD,      // palabra reservada "std"
    T_PRINTF,   // palabra reservada "printf"
    T_ENDL,     // palabra reservada "endl"
    T_MAIN,     // palabra reservada "main"
    
    // Identificadores y literales
    T_ID,       // identificador (e.g. variable)
    T_NUM,      // nÃºmero entero
    T_FLOAT_LIT,// nÃºmero flotante
    T_STRING,   // literal de cadena "..."
    T_CHAR_LIT, // literal de caracter '...'
    
    // Operadores
    T_ASSIGN,   // '='
    T_PLUS,     // '+'
    T_MINUS,    // '-'
    T_MULT,     // '*'
    T_DIV,      // '/'
    T_MOD,      // '%'
    T_EQ,       // '=='
    T_NEQ,      // '!='
    T_LT,       // '<'
    T_GT,       // '>'
    T_LE,       // '<='
    T_GE,       // '>='
    T_AND,      // '&&'
    T_OR,       // '||'
    T_NOT,      // '!'
    T_BITAND,   // '&'
    T_BITOR,    // '|'
    T_BITXOR,   // '^'
    T_SHL,      // '<<'
    T_SHR,      // '>>'
    T_INC,      // '++'
    T_DEC,      // '--'
    
    // Separadores
    T_SEMI,     // ';'
    T_COLON,    // ':'
    T_COMMA,    // ','
    T_DOT,      // '.'
    T_LPAREN,   // '('
    T_RPAREN,   // ')'
    T_LBRACE,   // '{'
    T_RBRACE,   // '}'
    T_LBRACKET, // '['
    T_RBRACKET, // ']'
    T_POUND,    // '#'
    T_DQUOTE,   // '"'
    
    // Especiales
    T_EOF,      // fin de la entrada
    T_UNKNOWN   // token no reconocido
};

// Estructura que representa un token
struct Token {
    TokenType type;
    string text; // texto original (por ejemplo, "abc", "123", etc.)
    int line;    // lÃ­nea de cÃ³digo donde aparece
    int column;  // columna donde aparece
};

// Mapa de palabras reservadas
static map<string, TokenType> KEYWORDS = {
    {"int", T_INT},
    {"float", T_FLOAT},
    {"char", T_CHAR},
    {"void", T_VOID},
    {"if", T_IF},
    {"else", T_ELSE},
    {"while", T_WHILE},
    {"for", T_FOR},
    {"return", T_RETURN},
    {"cout", T_COUT},
    {"cin", T_CIN},
    {"include", T_INCLUDE},
    {"using", T_USING},
    {"namespace", T_NAMESPACE},
    {"std", T_STD},
    {"printf", T_PRINTF},
    {"endl", T_ENDL}
};

// Analizador LÃ©xico (Scanner) mejorado
class Lexer {
private:
    string input;
    size_t pos;
    size_t length;
    int line;
    int column;

public:
    Lexer(const string &src) : input(src), pos(0), length(src.size()), line(1), column(1) {}

    // Avanza en el input mientras haya espacios, tabs o saltos de lÃ­nea
    void skipWhitespace() {
        while (pos < length && isspace(input[pos])) {
            if (input[pos] == '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
            pos++;
        }
    }
    
    // Salta comentarios de una lÃ­nea y multilÃ­nea
    void skipComments() {
        if (pos + 1 < length && input[pos] == '/' && input[pos + 1] == '/') {
            // Comentario de una lÃ­nea
            while (pos < length && input[pos] != '\n') {
                pos++;
                column++;
            }
        } else if (pos + 1 < length && input[pos] == '/' && input[pos + 1] == '*') {
            // Comentario multilÃ­nea
            pos += 2;
            column += 2;
            while (pos + 1 < length && !(input[pos] == '*' && input[pos + 1] == '/')) {
                if (input[pos] == '\n') {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
                pos++;
            }
            if (pos + 1 < length) {
                pos += 2;
                column += 2;
            }
        }
    }

    // Retorna el siguiente token sin consumirlo
    Token peekToken() {
        size_t oldPos = pos;
        int oldLine = line;
        int oldColumn = column;
        Token t = getNextToken();
        pos = oldPos;
        line = oldLine;
        column = oldColumn;
        return t;
    }

    // Obtiene el siguiente token y avanza el cursor
    Token getNextToken() {
        skipWhitespace();
        
        // Si estamos al inicio de un comentario, lo saltamos y obtenemos el siguiente token
        if (pos + 1 < length && ((input[pos] == '/' && input[pos + 1] == '/') || 
                                 (input[pos] == '/' && input[pos + 1] == '*'))) {
            skipComments();
            return getNextToken();
        }
        
        if (pos >= length) {
            return Token{ T_EOF, "", line, column };
        }

        int startColumn = column;
        char c = input[pos];

        // Palabras reservadas o identificadores
        if (isalpha(c) || c == '_') {
            size_t start = pos;
            while (pos < length && (isalnum(input[pos]) || input[pos] == '_')) {
                pos++;
                column++;
            }
            string text = input.substr(start, pos - start);

            // Verificar si es una palabra reservada
            auto it = KEYWORDS.find(text);
            if (it != KEYWORDS.end()) {
                return Token{ it->second, text, line, startColumn };
            } else {
                return Token{ T_ID, text, line, startColumn };
            }
        }

        // NÃºmeros (enteros o flotantes)
        if (isdigit(c)) {
            size_t start = pos;
            bool isFloat = false;
            
            // Parte entera
            while (pos < length && isdigit(input[pos])) {
                pos++;
                column++;
            }
            
            // Parte decimal si existe
            if (pos < length && input[pos] == '.') {
                isFloat = true;
                pos++;
                column++;
                
                while (pos < length && isdigit(input[pos])) {
                    pos++;
                    column++;
                }
            }
            
            // Parte exponencial si existe (e.g., 1e10, 1.5e-3)
            if (pos < length && (input[pos] == 'e' || input[pos] == 'E')) {
                isFloat = true;
                pos++;
                column++;
                
                // Signo opcional del exponente
                if (pos < length && (input[pos] == '+' || input[pos] == '-')) {
                    pos++;
                    column++;
                }
                
                // DÃ­gitos del exponente
                if (pos < length && isdigit(input[pos])) {
                    while (pos < length && isdigit(input[pos])) {
                        pos++;
                        column++;
                    }
                } else {
                    // Error: 'e' no seguido de dÃ­gitos
                    return Token{ T_UNKNOWN, input.substr(start, pos - start), line, startColumn };
                }
            }
            
            string text = input.substr(start, pos - start);
            return Token{ isFloat ? T_FLOAT_LIT : T_NUM, text, line, startColumn };
        }
        
        // Literales de cadena
        if (c == '"') {
            size_t start = pos;
            pos++;  // Saltar las comillas iniciales
            column++;
            
            while (pos < length && input[pos] != '"') {
                // Manejo de secuencias de escape
                if (input[pos] == '\\' && pos + 1 < length) {
                    pos += 2;  // Saltar el carÃ¡cter de escape y el siguiente
                    column += 2;
                } else {
                    if (input[pos] == '\n') {
                        line++;
                        column = 1;
                    } else {
                        column++;
                    }
                    pos++;
                }
            }
            
            if (pos < length) {
                pos++;  // Saltar las comillas finales
                column++;
            }
            
            string text = input.substr(start, pos - start);
            return Token{ T_STRING, text, line, startColumn };
        }
        
        // Literales de carÃ¡cter
        if (c == '\'') {
            size_t start = pos;
            pos++;  // Saltar la comilla simple inicial
            column++;
            
            // Manejar secuencias de escape
            if (pos < length && input[pos] == '\\') {
                pos += 2;  // Saltar la barra invertida y el carÃ¡cter de escape
                column += 2;
            } else if (pos < length) {
                pos++;
                column++;
            }
            
            // Debe haber una comilla simple de cierre
            if (pos < length && input[pos] == '\'') {
                pos++;
                column++;
            }
            
            string text = input.substr(start, pos - start);
            return Token{ T_CHAR_LIT, text, line, startColumn };
        }

        // Operadores de doble carÃ¡cter y otros tokens
        if (pos + 1 < length) {
            string op = input.substr(pos, 2);
            if (op == "==") { pos += 2; column += 2; return Token{ T_EQ, op, line, startColumn }; }
            if (op == "!=") { pos += 2; column += 2; return Token{ T_NEQ, op, line, startColumn }; }
            if (op == "<=") { pos += 2; column += 2; return Token{ T_LE, op, line, startColumn }; }
            if (op == ">=") { pos += 2; column += 2; return Token{ T_GE, op, line, startColumn }; }
            if (op == "&&") { pos += 2; column += 2; return Token{ T_AND, op, line, startColumn }; }
            if (op == "||") { pos += 2; column += 2; return Token{ T_OR, op, line, startColumn }; }
            if (op == "<<") { pos += 2; column += 2; return Token{ T_SHL, op, line, startColumn }; }
            if (op == ">>") { pos += 2; column += 2; return Token{ T_SHR, op, line, startColumn }; }
            if (op == "++") { pos += 2; column += 2; return Token{ T_INC, op, line, startColumn }; }
            if (op == "--") { pos += 2; column += 2; return Token{ T_DEC, op, line, startColumn }; }
        }

        // Caracteres simples
        pos++;
        column++;
        switch (c) {
            case '=': return Token{ T_ASSIGN, "=", line, startColumn };
            case ';': return Token{ T_SEMI, ";", line, startColumn };
            case ':': return Token{ T_COLON, ":", line, startColumn };
            case ',': return Token{ T_COMMA, ",", line, startColumn };
            case '.': return Token{ T_DOT, ".", line, startColumn };
            case '(': return Token{ T_LPAREN, "(", line, startColumn };
            case ')': return Token{ T_RPAREN, ")", line, startColumn };
            case '{': return Token{ T_LBRACE, "{", line, startColumn };
            case '}': return Token{ T_RBRACE, "}", line, startColumn };
            case '[': return Token{ T_LBRACKET, "[", line, startColumn };
            case ']': return Token{ T_RBRACKET, "]", line, startColumn };
            case '+': return Token{ T_PLUS, "+", line, startColumn };
            case '-': return Token{ T_MINUS, "-", line, startColumn };
            case '*': return Token{ T_MULT, "*", line, startColumn };
            case '/': return Token{ T_DIV, "/", line, startColumn };
            case '%': return Token{ T_MOD, "%", line, startColumn };
            case '<': return Token{ T_LT, "<", line, startColumn };
            case '>': return Token{ T_GT, ">", line, startColumn };
            case '!': return Token{ T_NOT, "!", line, startColumn };
            case '&': return Token{ T_BITAND, "&", line, startColumn };
            case '|': return Token{ T_BITOR, "|", line, startColumn };
            case '^': return Token{ T_BITXOR, "^", line, startColumn };
            case '#': return Token{ T_POUND, "#", line, startColumn };
            case '"': return Token{ T_DQUOTE, "\"", line, startColumn };
            default:  return Token{ T_UNKNOWN, string(1, c), line, startColumn };
        }
    }
};

/***************************************
 * 2. Estructuras para el AnÃ¡lisis SintÃ¡ctico (AST)
 ****************************************/

// Clase base para expresiones
class Expr {
public:
    virtual ~Expr() = default;
    virtual string getType() const { return "void"; } // Tipo por defecto
};

// ExpresiÃ³n para nÃºmeros enteros
class IntLiteralExpr : public Expr {
public:
    int value;
    IntLiteralExpr(int v) : value(v) {}
    string getType() const override { return "int"; }
};

// ExpresiÃ³n para nÃºmeros flotantes
class FloatLiteralExpr : public Expr {
public:
    double value;
    FloatLiteralExpr(double v) : value(v) {}
    string getType() const override { return "float"; }
};

// ExpresiÃ³n para caracteres
class CharLiteralExpr : public Expr {
public:
    char value;
    CharLiteralExpr(char v) : value(v) {}
    string getType() const override { return "char"; }
};

// ExpresiÃ³n para cadenas
class StringLiteralExpr : public Expr {
public:
    string value;
    StringLiteralExpr(const string &v) : value(v) {}
    string getType() const override { return "string"; }
};

// ExpresiÃ³n para variables
class VariableExpr : public Expr {
public:
    string name;
    string type;  // Tipo de la variable (int, float, etc.)
    
    VariableExpr(const string &n, const string &t = "unknown") 
        : name(n), type(t) {}
    
    string getType() const override { return type; }
};

// ExpresiÃ³n binaria (ej: a + 5)
class BinaryExpr : public Expr {
public:
    enum OpType {
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
        OP_EQ, OP_NEQ, OP_LT, OP_GT, OP_LE, OP_GE,
        OP_AND, OP_OR, OP_BITAND, OP_BITOR, OP_BITXOR,
        OP_SHL, OP_SHR
    };
    
    OpType op;
    unique_ptr<Expr> left;
    unique_ptr<Expr> right;
    
    BinaryExpr(OpType operation, Expr* l, Expr* r)
        : op(operation), left(l), right(r) {}
    
    string getType() const override {
        // Tipos de resultado para operadores binarios
        string leftType = left->getType();
        string rightType = right->getType();
        
        // Para operadores relacionales (==, !=, <, >, <=, >=)
        if (op == OP_EQ || op == OP_NEQ || op == OP_LT || op == OP_GT || op == OP_LE || op == OP_GE ||
            op == OP_AND || op == OP_OR) {
            return "bool";
        }
        
        // Para operadores aritmÃ©ticos, el tipo resultante depende de los operandos
        if (leftType == "float" || rightType == "float") {
            return "float";
        }
        
        return "int"; // Por defecto
    }
};

// ExpresiÃ³n unaria (ej: -5, !expresion)
class UnaryExpr : public Expr {
public:
    enum OpType {
        OP_NEG,  // NegaciÃ³n aritmÃ©tica (-)
        OP_NOT,  // NegaciÃ³n lÃ³gica (!)
        OP_PRE_INC,  // Pre-incremento (++x)
        OP_PRE_DEC,  // Pre-decremento (--x)
        OP_POST_INC, // Post-incremento (x++)
        OP_POST_DEC  // Post-decremento (x--)
    };
    
    OpType op;
    unique_ptr<Expr> operand;
    
    UnaryExpr(OpType operation, Expr* expr)
        : op(operation), operand(expr) {}
    
    string getType() const override {
        if (op == OP_NOT) {
            return "bool";
        }
        return operand->getType();
    }
};

// ExpresiÃ³n de llamada a funciÃ³n (ej: printf("hola"))
class CallExpr : public Expr {
public:
    string funcName;
    vector<unique_ptr<Expr>> args;
    string returnType;
    
    CallExpr(const string &name, string retType = "void") 
        : funcName(name), returnType(retType) {}
    
    void addArg(Expr* arg) {
        args.push_back(unique_ptr<Expr>(arg));
    }
    
    string getType() const override {
        return returnType;
    }
};

// Clase base para sentencias
class Stmt {
public:
    virtual ~Stmt() = default;
};

// DeclaraciÃ³n de variable: int a;
class VarDeclStmt : public Stmt {
public:
    string type;     // Tipo de la variable (int, float, etc.)
    string varName;  // Nombre de la variable
    unique_ptr<Expr> initializer; // ExpresiÃ³n de inicializaciÃ³n, si existe
    
    VarDeclStmt(const string &t, const string &name, Expr* init = nullptr)
        : type(t), varName(name), initializer(init) {}
};

// AsignaciÃ³n: a = <expr>;
class AssignStmt : public Stmt {
public:
    unique_ptr<Expr> lvalue;    // Lado izquierdo (generalmente una variable)
    unique_ptr<Expr> expression; // Lado derecho
    
    AssignStmt(Expr* lhs, Expr* rhs)
        : lvalue(lhs), expression(rhs) {}
};

// ExpresiÃ³n como sentencia (por ejemplo, llamadas a funciones): printf("hola");
class ExprStmt : public Stmt {
public:
    unique_ptr<Expr> expression;
    
    ExprStmt(Expr* expr) : expression(expr) {}
};

// Sentencia if: if (condiciÃ³n) {...} else {...}
class IfStmt : public Stmt {
public:
    unique_ptr<Expr> condition;
    unique_ptr<Stmt> thenBranch;
    unique_ptr<Stmt> elseBranch; // Puede ser nullptr
    
    IfStmt(Expr* cond, Stmt* then, Stmt* els = nullptr)
        : condition(cond), thenBranch(then), elseBranch(els) {}
};

// Sentencia while: while (condiciÃ³n) {...}
class WhileStmt : public Stmt {
public:
    unique_ptr<Expr> condition;
    unique_ptr<Stmt> body;
    
    WhileStmt(Expr* cond, Stmt* b)
        : condition(cond), body(b) {}
};

// Sentencia for: for (init; cond; incr) {...}
class ForStmt : public Stmt {
public:
    unique_ptr<Stmt> initializer; // Puede ser nullptr
    unique_ptr<Expr> condition;   // Puede ser nullptr
    unique_ptr<Expr> increment;   // Puede ser nullptr
    unique_ptr<Stmt> body;
    
    ForStmt(Stmt* init, Expr* cond, Expr* incr, Stmt* b)
        : initializer(init), condition(cond), increment(incr), body(b) {}
};

// Sentencia return: return <expr>;
class ReturnStmt : public Stmt {
public:
    unique_ptr<Expr> value; // Puede ser nullptr para return;
    
    ReturnStmt(Expr* val = nullptr) : value(val) {}
};

// Sentencia print: printf("...");
class PrintStmt : public Stmt {
public:
    string format;                  // Formato para printf
    vector<unique_ptr<Expr>> args;  // Argumentos para printf
    
    PrintStmt(const string &fmt) : format(fmt) {}
    
    void addArg(Expr* arg) {
        args.push_back(unique_ptr<Expr>(arg));
    }
};

// Sentencia para cin: cin >> var;
class CinStmt : public Stmt {
public:
    vector<unique_ptr<Expr>> variables; // Lista de variables a leer
    
    void addVariable(Expr* var) {
        variables.push_back(unique_ptr<Expr>(var));
    }
};

// Sentencia para cout: cout << expr << expr;
class CoutStmt : public Stmt {
public:
    vector<unique_ptr<Expr>> expressions; // Lista de expresiones a imprimir
    bool hasEndl;                         // Si termina con endl
    
    CoutStmt() : hasEndl(false) {}
    
    void addExpression(Expr* expr) {
        expressions.push_back(unique_ptr<Expr>(expr));
    }
    
    void setHasEndl(bool has_endl) {
        hasEndl = has_endl;
    }
};

// Bloque de cÃ³digo: { stmt1; stmt2; ... }
class BlockStmt : public Stmt {
public:
    vector<unique_ptr<Stmt>> statements;
    
    void addStatement(Stmt* stmt) {
        statements.push_back(unique_ptr<Stmt>(stmt));
    }
};

// DeclaraciÃ³n de funciÃ³n: tipo nombre(params) { cuerpo }
class FunctionDecl {
public:
    string returnType;               // Tipo de retorno
    string name;                     // Nombre de la funciÃ³n
    vector<VarDeclStmt*> parameters; // ParÃ¡metros
    unique_ptr<BlockStmt> body;      // Cuerpo de la funciÃ³n
    
    FunctionDecl(const string &ret, const string &n, BlockStmt* b = nullptr)
        : returnType(ret), name(n), body(b) {}
    
    ~FunctionDecl() {
        for (auto param : parameters) {
            delete param;
        }
    }
    
    void addParameter(VarDeclStmt* param) {
        parameters.push_back(param);
    }
};

// Directiva de inclusiÃ³n: #include <...>
class IncludeDirective {
public:
    string filename;
    bool isSystemHeader; // true para <...>, false para "..."
    
    IncludeDirective(const string &file, bool system)
        : filename(file), isSystemHeader(system) {}
};

// Programa completo
class Program {
public:
    vector<unique_ptr<IncludeDirective>> includes;  // Directivas #include
    vector<unique_ptr<FunctionDecl>> functions;     // Funciones globales
    
    void addInclude(IncludeDirective* include) {
        includes.push_back(unique_ptr<IncludeDirective>(include));
    }
    
    void addFunction(FunctionDecl* function) {
        functions.push_back(unique_ptr<FunctionDecl>(function));
    }
};

/***************************************
 * 3. Parser (AnÃ¡lisis SintÃ¡ctico y SemÃ¡ntico)
 ****************************************/

class Parser {
private:
    Lexer &lexer;
    Token currentToken;
    map<string, string> symbolTable; // Tabla de sÃ­mbolos: nombre -> tipo

    // FunciÃ³n auxiliar para consumir el token actual y avanzar
    void advance() {
        currentToken = lexer.getNextToken();
    }

    // Verifica que el token actual sea del tipo esperado; de lo contrario, lanza error
    void eat(TokenType type) {
        if (currentToken.type == type) {
            advance();
        } else {
            throw runtime_error("Error de sintaxis: se esperaba otro token, encontrado: " + 
                                currentToken.text + " en linea " + to_string(currentToken.line) + 
                                " columna " + to_string(currentToken.column));
        }
    }

    // <program> -> (<include>)* (<function_definition>)*
    unique_ptr<Program> parseProgram() {
        auto program = make_unique<Program>();
        
        // Procesar directivas de inclusiÃ³n
        while (currentToken.type == T_POUND) {
            advance();
            if (currentToken.type == T_INCLUDE) {
                auto include = parseInclude();
                program->addInclude(include);
            } else {
                throw runtime_error("Error de sintaxis: se esperaba 'include' despuÃ©s de '#'");
            }
        }
        
        // Procesar using namespace (opcional)
        if (currentToken.type == T_USING) {
            advance();
            eat(T_NAMESPACE);
            eat(T_STD);
            eat(T_SEMI);
        }
        
        // Procesar definiciones de funciones
        while (currentToken.type != T_EOF) {
            auto function = parseFunctionDefinition();
            program->addFunction(function);
        }
        
        return program;
    }
    
    // <include> -> '#' 'include' ('<' <filename> '>' | '"' <filename> '"')
    IncludeDirective* parseInclude() {
        eat(T_INCLUDE);
        
        bool isSystemHeader = false;
        string filename;
        
        if (currentToken.type == T_LT) {
            isSystemHeader = true;
            advance();
            
            // Recolectar el nombre del archivo hasta encontrar '>'
            stringstream filenameSS;
            while (currentToken.type != T_GT && currentToken.type != T_EOF) {
                filenameSS << currentToken.text;
                advance();
            }
            filename = filenameSS.str();
            eat(T_GT);
        } else if (currentToken.type == T_DQUOTE) {
            advance();
            
            // Recolectar el nombre del archivo hasta encontrar otra comilla
            stringstream filenameSS;
            while (currentToken.type != T_DQUOTE && currentToken.type != T_EOF) {
                filenameSS << currentToken.text;
                advance();
            }
            filename = filenameSS.str();
            eat(T_DQUOTE);
        } else {
            throw runtime_error("Error de sintaxis: se esperaba '<' o '\"' despuÃ©s de include");
        }
        
        return new IncludeDirective(filename, isSystemHeader);
    }
    
    // <function_definition> -> <type> <id> '(' <parameters> ')' <block>
    FunctionDecl* parseFunctionDefinition() {
        string returnType = parseType();
        
        // Verificamos si es un identificador o la palabra main
        if (currentToken.type != T_ID && currentToken.text != "main") {
            throw runtime_error("Error de sintaxis: se esperaba un identificador para nombre de funcion, encontrado: " + 
                                currentToken.text + " en linea " + to_string(currentToken.line) + 
                                " columna " + to_string(currentToken.column));
        }
        
        string name = currentToken.text;
        advance(); // Consumimos el identificador o 'main'
        
        eat(T_LPAREN);
        vector<VarDeclStmt*> parameters;
        
        // Procesar parÃ¡metros si hay alguno
        if (currentToken.type != T_RPAREN) {
            parameters = parseParameterList();
        }
        
        eat(T_RPAREN);
        
        BlockStmt* body = parseBlock();
        
        auto function = new FunctionDecl(returnType, name, body);
        for (auto param : parameters) {
            function->addParameter(param);
        }
        
        return function;
    }
    
    // <parameters> -> <parameter> (',' <parameter>)*
    vector<VarDeclStmt*> parseParameterList() {
        vector<VarDeclStmt*> parameters;
        
        // Primer parÃ¡metro
        parameters.push_back(parseParameter());
        
        // ParÃ¡metros adicionales
        while (currentToken.type == T_COMMA) {
            advance();
            parameters.push_back(parseParameter());
        }
        
        return parameters;
    }
    
    // <parameter> -> <type> <id>
    VarDeclStmt* parseParameter() {
        string type = parseType();
        string name = currentToken.text;
        eat(T_ID);
        
        symbolTable[name] = type;
        return new VarDeclStmt(type, name);
    }
    
    // <type> -> 'int' | 'float' | 'char' | 'void'
    string parseType() {
        string type;
        
        switch (currentToken.type) {
            case T_INT:
                type = "int";
                advance();
                break;
            case T_FLOAT:
                type = "float";
                advance();
                break;
            case T_CHAR:
                type = "char";
                advance();
                break;
            case T_VOID:
                type = "void";
                advance();
                break;
            default:
                throw runtime_error("Error de sintaxis: se esperaba un tipo");
        }
        
        return type;
    }
    
    // <block> -> '{' (<statement>)* '}'
    BlockStmt* parseBlock() {
        eat(T_LBRACE);
        
        auto block = new BlockStmt();
        
        while (currentToken.type != T_RBRACE && currentToken.type != T_EOF) {
            auto stmt = parseStatement();
            block->addStatement(stmt);
        }
        
        eat(T_RBRACE);
        return block;
    }
    
    // <statement> -> <var_decl> | <assign> | <if> | <while> | <for> | <return> | <expr_stmt> | <block>
    Stmt* parseStatement() {
        if (currentToken.type == T_INT || currentToken.type == T_FLOAT || currentToken.type == T_CHAR) {
            return parseVarDeclStatement();
        } else if (currentToken.type == T_IF) {
            return parseIfStatement();
        } else if (currentToken.type == T_WHILE) {
            return parseWhileStatement();
        } else if (currentToken.type == T_FOR) {
            return parseForStatement();
        } else if (currentToken.type == T_RETURN) {
            return parseReturnStatement();
        } else if (currentToken.type == T_LBRACE) {
            return parseBlock();
        } else if (currentToken.type == T_COUT) {
            return parseCoutStatement();
        } else if (currentToken.type == T_CIN) {
            return parseCinStatement();
        } else if (currentToken.type == T_PRINTF) {
            return parsePrintfStatement();
        } else if (currentToken.type == T_ID) {
            // Puede ser una asignaciÃ³n o una llamada a funciÃ³n
            Token idToken = currentToken;
            advance();
            
            if (currentToken.type == T_ASSIGN || currentToken.type == T_INC || currentToken.type == T_DEC) {
                // Es una asignaciÃ³n
                return parseAssignStatementWithId(idToken.text);
            } else {
                // Es una llamada a funciÃ³n
                return parseExprStatementWithId(idToken.text);
            }
        } else {
            throw runtime_error("Error de sintaxis: sentencia desconocida");
        }
    }
    
    // <var_decl> -> <type> <id> ('=' <expr>)? ';'
    Stmt* parseVarDeclStatement() {
        string type = parseType();
        string name = currentToken.text;
        eat(T_ID);
        
        if (symbolTable.find(name) != symbolTable.end()) {
            throw runtime_error("Error semantico: variable '" + name + "' ya declarada");
        }
        
        symbolTable[name] = type;
        
        Expr* initializer = nullptr;
        if (currentToken.type == T_ASSIGN) {
            advance();
            initializer = parseExpression();
            
            // ComprobaciÃ³n simple de tipos
            if (initializer->getType() != type && 
                !(type == "float" && initializer->getType() == "int")) {
                throw runtime_error("Error semantico: tipo incompatible en la inicializacion");
            }
        }
        
        eat(T_SEMI);
        return new VarDeclStmt(type, name, initializer);
    }
    
    // <assign> -> <id> '=' <expr> ';'
    Stmt* parseAssignStatementWithId(const string &varName) {
        if (symbolTable.find(varName) == symbolTable.end()) {
            throw runtime_error("Error semantico: variable '" + varName + "' no declarada");
        }
        
        Expr* lvalue = new VariableExpr(varName, symbolTable[varName]);
        
        if (currentToken.type == T_INC) {
            // x++
            advance();
            eat(T_SEMI);
            return new ExprStmt(new UnaryExpr(UnaryExpr::OP_POST_INC, lvalue));
        } else if (currentToken.type == T_DEC) {
            // x--
            advance();
            eat(T_SEMI);
            return new ExprStmt(new UnaryExpr(UnaryExpr::OP_POST_DEC, lvalue));
        }
        
        eat(T_ASSIGN);
        Expr* rvalue = parseExpression();
        
        // ComprobaciÃ³n simple de tipos
        if (lvalue->getType() != rvalue->getType() && 
            !(lvalue->getType() == "float" && rvalue->getType() == "int")) {
            throw runtime_error("Error semantico: tipos incompatibles en la asignacion");
        }
        
        eat(T_SEMI);
        return new AssignStmt(lvalue, rvalue);
    }
    
    // <expr_stmt> -> <expr> ';'
    Stmt* parseExprStatementWithId(const string &funcName) {
        auto callExpr = parseCallExprWithId(funcName);
        eat(T_SEMI);
        return new ExprStmt(callExpr);
    }
    
    // <if> -> 'if' '(' <expr> ')' <statement> ('else' <statement>)?
    Stmt* parseIfStatement() {
        eat(T_IF);
        eat(T_LPAREN);
        Expr* condition = parseExpression();
        eat(T_RPAREN);
        
        Stmt* thenBranch = parseStatement();
        
        Stmt* elseBranch = nullptr;
        if (currentToken.type == T_ELSE) {
            advance();
            elseBranch = parseStatement();
        }
        
        return new IfStmt(condition, thenBranch, elseBranch);
    }
    
    // <while> -> 'while' '(' <expr> ')' <statement>
    Stmt* parseWhileStatement() {
        eat(T_WHILE);
        eat(T_LPAREN);
        Expr* condition = parseExpression();
        eat(T_RPAREN);
        
        Stmt* body = parseStatement();
        
        return new WhileStmt(condition, body);
    }
    
    // <for> -> 'for' '(' <for_init> ';' <expr>? ';' <expr>? ')' <statement>
    Stmt* parseForStatement() {
        eat(T_FOR);
        eat(T_LPAREN);
        
        // Inicializador
        Stmt* initializer = nullptr;
        if (currentToken.type != T_SEMI) {
            if (currentToken.type == T_INT || currentToken.type == T_FLOAT || currentToken.type == T_CHAR) {
                initializer = parseVarDeclStatement();
            } else {
                string varName = currentToken.text;
                eat(T_ID);
                initializer = parseAssignStatementWithId(varName);
            }
        } else {
            eat(T_SEMI);
        }
        
        // CondiciÃ³n
        Expr* condition = nullptr;
        if (currentToken.type != T_SEMI) {
            condition = parseExpression();
        }
        eat(T_SEMI);
        
        // Incremento
        Expr* increment = nullptr;
        if (currentToken.type != T_RPAREN) {
            increment = parseExpression();
        }
        eat(T_RPAREN);
        
        // Cuerpo
        Stmt* body = parseStatement();
        
        return new ForStmt(initializer, condition, increment, body);
    }
    
    // <return> -> 'return' <expr>? ';'
    Stmt* parseReturnStatement() {
        eat(T_RETURN);
        
        Expr* value = nullptr;
        if (currentToken.type != T_SEMI) {
            value = parseExpression();
        }
        
        eat(T_SEMI);
        return new ReturnStmt(value);
    }
    
    // <printf> -> 'printf' '(' <string> (',' <expr>)* ')' ';'
    Stmt* parsePrintfStatement() {
        eat(T_PRINTF);
        eat(T_LPAREN);
        
        if (currentToken.type != T_STRING) {
            throw runtime_error("Error de sintaxis: se esperaba una cadena de formato para printf");
        }
        
        string format = currentToken.text;
        advance();
        
        auto printStmt = new PrintStmt(format);
        
        while (currentToken.type == T_COMMA) {
            advance();
            Expr* arg = parseExpression();
            printStmt->addArg(arg);
        }
        
        eat(T_RPAREN);
        eat(T_SEMI);
        
        return printStmt;
    }
    
    // <cin> -> 'cin' '>>' <id> ('>>' <id>)* ';'
    Stmt* parseCinStatement() {
        eat(T_CIN);
        
        auto cinStmt = new CinStmt();
        
        do {
            eat(T_SHR);
            
            if (currentToken.type != T_ID) {
                throw runtime_error("Error de sintaxis: se esperaba un identificador despuÃ©s de '>>'");
            }
            
            string varName = currentToken.text;
            eat(T_ID);
            
            if (symbolTable.find(varName) == symbolTable.end()) {
                throw runtime_error("Error semantico: variable '" + varName + "' no declarada");
            }
            
            cinStmt->addVariable(new VariableExpr(varName, symbolTable[varName]));
            
        } while (currentToken.type == T_SHR);
        
        eat(T_SEMI);
        return cinStmt;
    }
    
    // <cout> -> 'cout' ('<<' <expr>)+ ('<<' 'endl')? ';'
    Stmt* parseCoutStatement() {
        eat(T_COUT);
        
        auto coutStmt = new CoutStmt();
        
        do {
            eat(T_SHL);
            
            if (currentToken.type == T_ENDL) {
                coutStmt->setHasEndl(true);
                advance();
                break;
            }
            
            Expr* expr = parseExpression();
            coutStmt->addExpression(expr);
            
        } while (currentToken.type == T_SHL);
        
        eat(T_SEMI);
        return coutStmt;
    }
    
    // <expr> -> <assignment>
    Expr* parseExpression() {
        return parseAssignment();
    }
    
    // <assignment> -> <logic_or> ('=' <assignment>)?
    Expr* parseAssignment() {
        Expr* expr = parseLogicOr();
        
        if (currentToken.type == T_ASSIGN) {
            advance();
            Expr* value = parseAssignment();
            
            // Solo las variables pueden ser asignadas
            auto var = dynamic_cast<VariableExpr*>(expr);
            if (!var) {
                throw runtime_error("Error semantico: el lado izquierdo de una asignacion debe ser una variable");
            }
            
            // Verificar tipos
            if (var->getType() != value->getType() && 
                !(var->getType() == "float" && value->getType() == "int")) {
                throw runtime_error("Error semantico: tipos incompatibles en la asignacion");
            }
            
            delete expr; // Ya no necesitamos la expresiÃ³n original
            
            // Creamos una nueva variable y la devolvemos (para expresiones como a = b = c)
            return new VariableExpr(var->name, var->type);
        }
        
        return expr;
    }
    
    // <logic_or> -> <logic_and> ('||' <logic_and>)*
    Expr* parseLogicOr() {
        Expr* expr = parseLogicAnd();
        
        while (currentToken.type == T_OR) {
            advance();
            Expr* right = parseLogicAnd();
            expr = new BinaryExpr(BinaryExpr::OP_OR, expr, right);
        }
        
        return expr;
    }
    
    // <logic_and> -> <equality> ('&&' <equality>)*
    Expr* parseLogicAnd() {
        Expr* expr = parseEquality();
        
        while (currentToken.type == T_AND) {
            advance();
            Expr* right = parseEquality();
            expr = new BinaryExpr(BinaryExpr::OP_AND, expr, right);
        }
        
        return expr;
    }
    
    // <equality> -> <relational> (('==' | '!=') <relational>)*
    Expr* parseEquality() {
        Expr* expr = parseRelational();
        
        while (currentToken.type == T_EQ || currentToken.type == T_NEQ) {
            BinaryExpr::OpType op = (currentToken.type == T_EQ) ? BinaryExpr::OP_EQ : BinaryExpr::OP_NEQ;
            advance();
            Expr* right = parseRelational();
            expr = new BinaryExpr(op, expr, right);
        }
        
        return expr;
    }
    
    // <relational> -> <additive> (('<' | '>' | '<=' | '>=') <additive>)*
    Expr* parseRelational() {
        Expr* expr = parseAdditive();
        
        while (currentToken.type == T_LT || currentToken.type == T_GT || 
               currentToken.type == T_LE || currentToken.type == T_GE) {
            BinaryExpr::OpType op;
            
            switch (currentToken.type) {
                case T_LT: op = BinaryExpr::OP_LT; break;
                case T_GT: op = BinaryExpr::OP_GT; break;
                case T_LE: op = BinaryExpr::OP_LE; break;
                case T_GE: op = BinaryExpr::OP_GE; break;
                default: throw runtime_error("Operador relacional desconocido");
            }
            
            advance();
            Expr* right = parseAdditive();
            expr = new BinaryExpr(op, expr, right);
        }
        
        return expr;
    }
    
    // <additive> -> <multiplicative> (('+' | '-') <multiplicative>)*
    Expr* parseAdditive() {
        Expr* expr = parseMultiplicative();
        
        while (currentToken.type == T_PLUS || currentToken.type == T_MINUS) {
            BinaryExpr::OpType op = (currentToken.type == T_PLUS) ? BinaryExpr::OP_ADD : BinaryExpr::OP_SUB;
            advance();
            Expr* right = parseMultiplicative();
            expr = new BinaryExpr(op, expr, right);
        }
        
        return expr;
    }
    
    // <multiplicative> -> <unary> (('*' | '/' | '%') <unary>)*
    Expr* parseMultiplicative() {
        Expr* expr = parseUnary();
        
        while (currentToken.type == T_MULT || currentToken.type == T_DIV || currentToken.type == T_MOD) {
            BinaryExpr::OpType op;
            
            switch (currentToken.type) {
                case T_MULT: op = BinaryExpr::OP_MUL; break;
                case T_DIV: op = BinaryExpr::OP_DIV; break;
                case T_MOD: op = BinaryExpr::OP_MOD; break;
                default: throw runtime_error("Operador multiplicativo desconocido");
            }
            
            advance();
            Expr* right = parseUnary();
            expr = new BinaryExpr(op, expr, right);
        }
        
        return expr;
    }
    
    // <unary> -> ('!' | '-' | '++' | '--') <unary> | <primary>
    Expr* parseUnary() {
        if (currentToken.type == T_NOT) {
            advance();
            return new UnaryExpr(UnaryExpr::OP_NOT, parseUnary());
        } else if (currentToken.type == T_MINUS) {
            advance();
            return new UnaryExpr(UnaryExpr::OP_NEG, parseUnary());
        } else if (currentToken.type == T_INC) {
            advance();
            
            // Verificar que el operando sea una variable
            Expr* operand = parseUnary();
            auto var = dynamic_cast<VariableExpr*>(operand);
            if (!var) {
                throw runtime_error("Error semantico: el operando de ++ debe ser una variable");
            }
            
            return new UnaryExpr(UnaryExpr::OP_PRE_INC, operand);
        } else if (currentToken.type == T_DEC) {
            advance();
            
            // Verificar que el operando sea una variable
            Expr* operand = parseUnary();
            auto var = dynamic_cast<VariableExpr*>(operand);
            if (!var) {
                throw runtime_error("Error semantico: el operando de -- debe ser una variable");
            }
            
            return new UnaryExpr(UnaryExpr::OP_PRE_DEC, operand);
        } else {
            return parsePrimary();
        }
    }
    
    // <primary> -> <number> | <string> | <char> | <id> | '(' <expr> ')' | <call>
    Expr* parsePrimary() {
        if (currentToken.type == T_NUM) {
            int value = stoi(currentToken.text);
            advance();
            return new IntLiteralExpr(value);
        } else if (currentToken.type == T_FLOAT_LIT) {
            double value = stod(currentToken.text);
            advance();
            return new FloatLiteralExpr(value);
        } else if (currentToken.type == T_STRING) {
            string value = currentToken.text;
            advance();
            return new StringLiteralExpr(value);
        } else if (currentToken.type == T_CHAR_LIT) {
            char value = currentToken.text[1]; // Ignorar las comillas simples
            advance();
            return new CharLiteralExpr(value);
        } else if (currentToken.type == T_ID) {
            string name = currentToken.text;
            advance();
            
            // Verificar si es una llamada a funciÃ³n
            if (currentToken.type == T_LPAREN) {
                return parseCallExprWithId(name);
            }
            
            // Verificar si la variable estÃ¡ declarada
            if (symbolTable.find(name) == symbolTable.end()) {
                throw runtime_error("Error semantico: variable '" + name + "' no declarada");
            }
            
            // Verificar si es incremento/decremento postfijo
            if (currentToken.type == T_INC) {
                advance();
                return new UnaryExpr(UnaryExpr::OP_POST_INC, new VariableExpr(name, symbolTable[name]));
            } else if (currentToken.type == T_DEC) {
                advance();
                return new UnaryExpr(UnaryExpr::OP_POST_DEC, new VariableExpr(name, symbolTable[name]));
            }
            
            return new VariableExpr(name, symbolTable[name]);
        } else if (currentToken.type == T_LPAREN) {
            advance();
            Expr* expr = parseExpression();
            eat(T_RPAREN);
            return expr;
        } else {
            throw runtime_error("Error de sintaxis: expresion primaria inesperada");
        }
    }
    
    // <call> -> <id> '(' <arguments>? ')'
    CallExpr* parseCallExprWithId(const string &funcName) {
        // Determinamos el tipo de retorno (esto es simplificado, en un compilador real
        // se usarÃ­a una tabla de sÃ­mbolos de funciones)
        string returnType = "void";
        if (funcName == "printf") {
            returnType = "int";
        }
        
        auto call = new CallExpr(funcName, returnType);
        
        eat(T_LPAREN);
        
        // Procesar argumentos si hay alguno
        if (currentToken.type != T_RPAREN) {
            parseArguments(call);
        }
        
        eat(T_RPAREN);
        return call;
    }
    
    // <arguments> -> <expr> (',' <expr>)*
    void parseArguments(CallExpr* call) {
        call->addArg(parseExpression());
        
        while (currentToken.type == T_COMMA) {
            advance();
            call->addArg(parseExpression());
        }
    }

public:
    Parser(Lexer &l) : lexer(l) {
        advance();
    }

    unique_ptr<Program> parse() {
        return parseProgram();
    }
};

/***************************************
 * 4. GeneraciÃ³n de CÃ³digo Intermedio (CÃ³digo de Tres Direcciones)
 ****************************************/

class CodeGenerator {
private:
    int tempCount;
    int labelCount;
    
    // Genera un nombre nuevo para variables temporales
    string newTemp() {
        return "t" + to_string(tempCount++);
    }
    
    // Genera una etiqueta nueva
    string newLabel() {
        return "L" + to_string(labelCount++);
    }

public:
    CodeGenerator() : tempCount(0), labelCount(0) {}

    // Genera cÃ³digo para una expresiÃ³n y retorna el nombre de la variable donde se guarda el resultado
    string genExpr(Expr* expr, vector<string> &code) {
        if (auto num = dynamic_cast<IntLiteralExpr*>(expr)) {
            string t = newTemp();
            code.push_back(t + " = " + to_string(num->value));
            return t;
        } else if (auto fnum = dynamic_cast<FloatLiteralExpr*>(expr)) {
            string t = newTemp();
            code.push_back(t + " = " + to_string(fnum->value));
            return t;
        } else if (auto ch = dynamic_cast<CharLiteralExpr*>(expr)) {
            string t = newTemp();
            code.push_back(t + " = '" + string(1, ch->value) + "'");
            return t;
        } else if (auto str = dynamic_cast<StringLiteralExpr*>(expr)) {
            string t = newTemp();
            code.push_back(t + " = " + str->value);
            return t;
        } else if (auto var = dynamic_cast<VariableExpr*>(expr)) {
            return var->name;
        } else if (auto bin = dynamic_cast<BinaryExpr*>(expr)) {
            string left = genExpr(bin->left.get(), code);
            string right = genExpr(bin->right.get(), code);
            string t = newTemp();
            
            string opStr;
            switch (bin->op) {
                case BinaryExpr::OP_ADD: opStr = "+"; break;
                case BinaryExpr::OP_SUB: opStr = "-"; break;
                case BinaryExpr::OP_MUL: opStr = "*"; break;
                case BinaryExpr::OP_DIV: opStr = "/"; break;
                case BinaryExpr::OP_MOD: opStr = "%"; break;
                case BinaryExpr::OP_EQ: opStr = "=="; break;
                case BinaryExpr::OP_NEQ: opStr = "!="; break;
                case BinaryExpr::OP_LT: opStr = "<"; break;
                case BinaryExpr::OP_GT: opStr = ">"; break;
                case BinaryExpr::OP_LE: opStr = "<="; break;
                case BinaryExpr::OP_GE: opStr = ">="; break;
                case BinaryExpr::OP_AND: opStr = "&&"; break;
                case BinaryExpr::OP_OR: opStr = "||"; break;
                case BinaryExpr::OP_BITAND: opStr = "&"; break;
                case BinaryExpr::OP_BITOR: opStr = "|"; break;
                case BinaryExpr::OP_BITXOR: opStr = "^"; break;
                case BinaryExpr::OP_SHL: opStr = "<<"; break;
                case BinaryExpr::OP_SHR: opStr = ">>"; break;
                default: throw runtime_error("Operador binario desconocido");
            }
            
            code.push_back(t + " = " + left + " " + opStr + " " + right);
            return t;
        } else if (auto un = dynamic_cast<UnaryExpr*>(expr)) {
            string operand = genExpr(un->operand.get(), code);
            string t = newTemp();
            
            switch (un->op) {
                case UnaryExpr::OP_NEG:
                    code.push_back(t + " = -" + operand);
                    break;
                case UnaryExpr::OP_NOT:
                    code.push_back(t + " = !" + operand);
                    break;
                case UnaryExpr::OP_PRE_INC:
                    code.push_back(operand + " = " + operand + " + 1");
                    code.push_back(t + " = " + operand);
                    break;
                case UnaryExpr::OP_PRE_DEC:
                    code.push_back(operand + " = " + operand + " - 1");
                    code.push_back(t + " = " + operand);
                    break;
                case UnaryExpr::OP_POST_INC:
                    code.push_back(t + " = " + operand);
                    code.push_back(operand + " = " + operand + " + 1");
                    break;
                case UnaryExpr::OP_POST_DEC:
                    code.push_back(t + " = " + operand);
                    code.push_back(operand + " = " + operand + " - 1");
                    break;
                default:
                    throw runtime_error("Operador unario desconocido");
            }
            
            return t;
        } else if (auto call = dynamic_cast<CallExpr*>(expr)) {
            vector<string> args;
            for (const auto &arg : call->args) {
                args.push_back(genExpr(arg.get(), code));
            }
            
            string t = newTemp();
            
            // Construir la llamada
            stringstream ss;
            ss << t << " = CALL " << call->funcName << "(";
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << args[i];
            }
            ss << ")";
            
            code.push_back(ss.str());
            return t;
        }
        
        throw runtime_error("Expresion desconocida en generacion");
    }

    // Genera cÃ³digo para una sentencia
    void genStmt(Stmt* stmt, vector<string> &code) {
        if (auto decl = dynamic_cast<VarDeclStmt*>(stmt)) {
            code.push_back("DECL " + decl->type + " " + decl->varName);
            
            if (decl->initializer) {
                string init = genExpr(decl->initializer.get(), code);
                code.push_back(decl->varName + " = " + init);
            }
        } else if (auto assign = dynamic_cast<AssignStmt*>(stmt)) {
            string lhs = genExpr(assign->lvalue.get(), code);
            string rhs = genExpr(assign->expression.get(), code);
            code.push_back(lhs + " = " + rhs);
        } else if (auto exprStmt = dynamic_cast<ExprStmt*>(stmt)) {
            genExpr(exprStmt->expression.get(), code);
        } else if (auto ifStmt = dynamic_cast<IfStmt*>(stmt)) {
            string cond = genExpr(ifStmt->condition.get(), code);
            string elseLabel = newLabel();
            string endLabel = newLabel();
            
            code.push_back("IF NOT " + cond + " GOTO " + elseLabel);
            
            genStmt(ifStmt->thenBranch.get(), code);
            code.push_back("GOTO " + endLabel);
            
            code.push_back(elseLabel + ":");
            if (ifStmt->elseBranch) {
                genStmt(ifStmt->elseBranch.get(), code);
            }
            
            code.push_back(endLabel + ":");
        } else if (auto whileStmt = dynamic_cast<WhileStmt*>(stmt)) {
            string startLabel = newLabel();
            string endLabel = newLabel();
            
            code.push_back(startLabel + ":");
            string cond = genExpr(whileStmt->condition.get(), code);
            code.push_back("IF NOT " + cond + " GOTO " + endLabel);
            
            genStmt(whileStmt->body.get(), code);
            code.push_back("GOTO " + startLabel);
            
            code.push_back(endLabel + ":");
        } else if (auto forStmt = dynamic_cast<ForStmt*>(stmt)) {
            string startLabel = newLabel();
            string endLabel = newLabel();
            string updateLabel = newLabel();
            
            // Inicializador
            if (forStmt->initializer) {
                genStmt(forStmt->initializer.get(), code);
            }
            
            code.push_back(startLabel + ":");
            
            // CondiciÃ³n
            if (forStmt->condition) {
                string cond = genExpr(forStmt->condition.get(), code);
                code.push_back("IF NOT " + cond + " GOTO " + endLabel);
            }
            
            // Cuerpo
            genStmt(forStmt->body.get(), code);
            
            code.push_back(updateLabel + ":");
            
            // Incremento
            if (forStmt->increment) {
                genExpr(forStmt->increment.get(), code);
            }
            
            code.push_back("GOTO " + startLabel);
            code.push_back(endLabel + ":");
        } else if (auto returnStmt = dynamic_cast<ReturnStmt*>(stmt)) {
            if (returnStmt->value) {
                string val = genExpr(returnStmt->value.get(), code);
                code.push_back("RETURN " + val);
            } else {
                code.push_back("RETURN");
            }
        } else if (auto printStmt = dynamic_cast<PrintStmt*>(stmt)) {
            vector<string> args;
            for (const auto &arg : printStmt->args) {
                args.push_back(genExpr(arg.get(), code));
            }
            
            stringstream ss;
            ss << "PRINT " << printStmt->format;
            for (const auto &arg : args) {
                ss << ", " << arg;
            }
            
            code.push_back(ss.str());
        } else if (auto cinStmt = dynamic_cast<CinStmt*>(stmt)) {
            for (const auto &var : cinStmt->variables) {
                string varName = genExpr(var.get(), code);
                code.push_back("READ " + varName);
            }
        } else if (auto coutStmt = dynamic_cast<CoutStmt*>(stmt)) {
            for (const auto &expr : coutStmt->expressions) {
                string val = genExpr(expr.get(), code);
                code.push_back("WRITE " + val);
            }
            
            if (coutStmt->hasEndl) {
                code.push_back("WRITELN");
            }
        } else if (auto blockStmt = dynamic_cast<BlockStmt*>(stmt)) {
            code.push_back("// Begin Block");
            for (const auto &s : blockStmt->statements) {
                genStmt(s.get(), code);
            }
            code.push_back("// End Block");
        } else {
            throw runtime_error("Sentencia desconocida en generaciÃ³n");
        }
    }

    // Genera cÃ³digo para una funciÃ³n
    void genFunction(FunctionDecl* func, vector<string> &code) {
        code.push_back("FUNCTION " + func->returnType + " " + func->name + ":");
        
        // ParÃ¡metros
        for (auto param : func->parameters) {
            code.push_back("PARAM " + param->type + " " + param->varName);
        }
        
        // Cuerpo
        if (func->body) {
            genStmt(func->body.get(), code);
        }
        
        // Si es void y no tiene return explÃ­cito, aÃ±adimos uno
        if (func->returnType == "void") {
            code.push_back("RETURN");
        }
        
        code.push_back("END " + func->name);
    }

    // Genera cÃ³digo para una directiva include
    void genInclude(IncludeDirective* include, vector<string> &code) {
        if (include->isSystemHeader) {
            code.push_back("INCLUDE <" + include->filename + ">");
        } else {
            code.push_back("INCLUDE \"" + include->filename + "\"");
        }
    }

    // Genera cÃ³digo para todo el programa
    vector<string> generate(Program* program) {
        vector<string> code;
        
        // Directivas include
        for (const auto &include : program->includes) {
            genInclude(include.get(), code);
        }
        
        // Funciones
        for (const auto &func : program->functions) {
            genFunction(func.get(), code);
        }
        
        return code;
    }
};

/***************************************
 * 5. EvaluaciÃ³n y EjecuciÃ³n
 ****************************************/

// Esta clase es una versiÃ³n simplificada de un intÃ©rprete
// para el cÃ³digo intermedio, solo para demostrar la funcionalidad
class Interpreter {
private:
    map<string, int> variables;
    vector<string> output;
    
    int evaluateExpr(const string &expr) {
        // Este es un interpretador extremadamente simplificado
        // Solo maneja variables ya evaluadas y literales enteras
        
        // Verificar si es una variable
        if (variables.find(expr) != variables.end()) {
            return variables[expr];
        }
        
        // Intentar convertir a entero
        try {
            return stoi(expr);
        } catch (...) {
            // No es un entero
        }
        
        // No podemos manejar expresiones complejas en esta versiÃ³n simplificada
        throw runtime_error("No se puede evaluar la expresiÃ³n: " + expr);
    }
    
public:
    void execute(const vector<string> &code) {
        for (const auto &line : code) {
            if (line.find("DECL") == 0 || line.find("//") == 0 || 
                line.find("FUNCTION") == 0 || line.find("END") == 0 ||
                line.find("PARAM") == 0 || line.find("INCLUDE") == 0) {
                // Ignorar declaraciones, comentarios, declaraciones de funciÃ³n, etc.
                continue;
            } else if (line.find("PRINT") == 0) {
                // Manejar impresiÃ³n
                size_t start = line.find("\"");
                size_t end = line.find("\"", start + 1);
                
                if (start != string::npos && end != string::npos) {
                    string format = line.substr(start + 1, end - start - 1);
                    output.push_back(format);
                }
            } else if (line.find("READ") == 0) {
                // SimulaciÃ³n de lectura
                string var = line.substr(5);
                cout << "Simulando lectura para " << var << ". Ingrese un valor: ";
                int value;
                cin >> value;
                variables[var] = value;
            } else if (line.find("WRITE") == 0) {
                // Manejar escritura
                string var = line.substr(6);
                try {
                    int value = evaluateExpr(var);
                    output.push_back(to_string(value));
                } catch (...) {
                    output.push_back(var); // Si no podemos evaluar, mostramos la variable como texto
                }
            } else if (line.find("WRITELN") == 0) {
                // Salto de lÃ­nea
                output.push_back("\n");
            } else if (line.find(" = ") != string::npos) {
                // AsignaciÃ³n
                size_t pos = line.find(" = ");
                string var = line.substr(0, pos);
                string expr = line.substr(pos + 3);
                
                try {
                    int value = evaluateExpr(expr);
                    variables[var] = value;
                } catch (...) {
                    // Si no podemos evaluar, ignoramos
                }
            }
            // Otras instrucciones (if, goto, etc.) no se manejan en esta versiÃ³n simplificada
        }
    }
    
    const vector<string>& getOutput() const {
        return output;
    }
};

/***************************************
 * 6. FunciÃ³n Principal
 ****************************************/

int main() {
    cout << "========================\n";
    cout << " Compilador de C++ Simplificado\n";
    cout << "========================\n\n";

    cout << "Ingresa el codigo fuente C++ (finaliza con Ctrl+Z en una linea vacia y luego ENTER):\n\n";
    
    // MÃ©todo alternativo para leer la entrada con mejor detecciÃ³n de Ctrl+Z en Windows
    string sourceCode = "";
    string line;
    
    cout << "> "; // Indicador de entrada
    while (true) {
        if (!getline(cin, line)) {
            // Si getline devuelve false, se ha detectado EOF (Ctrl+Z)
            break;
        }
        sourceCode += line + "\n";
        cout << "> "; // Mostrar indicador para cada nueva lÃ­nea
    }
    
    // Si no hay entrada o solo se presionÃ³ Ctrl+Z sin cÃ³digo
    if (sourceCode.empty() || sourceCode == "\n") {
        cerr << "\nNo se proporciono codigo fuente. Saliendo...\n";
        return 1;
    }
    
    cout << "\nCodigo fuente recibido correctamente. Procesando...\n";
    
    try {
        // 1. AnÃ¡lisis lÃ©xico
        Lexer lexer(sourceCode);
        
        // 2. AnÃ¡lisis sintÃ¡ctico
        Parser parser(lexer);
        auto program = parser.parse();
        
        // 3. GeneraciÃ³n de cÃ³digo intermedio
        CodeGenerator codeGen;
        vector<string> intermediateCode = codeGen.generate(program.get());
        
        cout << "\n=== Codigo Intermedio ===\n";
        for (const auto &line : intermediateCode) {
            cout << line << "\n";
        }
        
        // 4. InterpretaciÃ³n/EjecuciÃ³n simplificada (opcional)
        cout << "\n=== Ejecucion Simulada ===\n";
        Interpreter interpreter;
        interpreter.execute(intermediateCode);
        
        for (const auto &output : interpreter.getOutput()) {
            cout << output;
        }
        
    } catch (const exception &ex) {
        cerr << "\nERROR: " << ex.what() << endl;
        return 1;
    }
    
    cout << "\n=== Compilacion y ejecucion completada ===\n";
    return 0;
}
