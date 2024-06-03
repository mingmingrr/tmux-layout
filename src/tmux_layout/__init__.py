#!/usr/bin/env python3

from __future__ import annotations

from typing import TypeVar, Optional, TypeAlias, Iterator, ClassVar, \
	overload, Sequence, Iterable, Union, Callable, Generic

from fractions import Fraction
import time
import abc
import enum
import dataclasses
import logging
import sys
import libtmux

T = TypeVar('T')
def check(x: Optional[T]) -> T:
	assert x is not None
	return x

def splitline(xs:str) -> tuple[str,str]:
	n, *ns = xs.split('\n', 1)
	return n, ns[0] if ns else ''

@dataclasses.dataclass
class Vec2d:
	x: int = 0
	y: int = 0

	def __add__(self, other:Union[Vec2d,tuple[int,int]]) -> Vec2d:
		if isinstance(other, tuple): other = Vec2d(*other)
		return Vec2d(self.x + other.x, self.y + other.y)

	def __mul__(self, other:Union[int,Vec2d,tuple[int,int]]) -> Vec2d:
		if isinstance(other, int): other = Vec2d(other, other)
		elif isinstance(other, tuple): other = Vec2d(*other)
		return Vec2d(self.x * other.x, self.y * other.y)

	def mirror(self) -> Vec2d:
		return Vec2d(self.y, self.x)

class Direction(enum.Enum):
	Horizontal = Vec2d(1, 0)
	Vertical = Vec2d(0, 1)

	def mirror(self) -> Direction:
		if self == Direction.Horizontal:
			return Direction.Vertical
		return Direction.Horizontal

PaneID: TypeAlias = str
WindowID: TypeAlias = str
WindowIndex: TypeAlias = int
SessionID: TypeAlias = str
SessionName: TypeAlias = str
ServerPath: TypeAlias = str

class Geometry(metaclass=abc.ABCMeta):
	@staticmethod
	def checksum(string:str):
		c = 0
		for n in string:
			c = (((c & 1) << 15) + (c >> 1) + ord(n)) & 0xffff
		return c

	@abc.abstractmethod
	def iter(self) -> Iterator[PaneID]: ...

	@abc.abstractmethod
	def mirror(self) -> Geometry: ...

	@abc.abstractmethod
	def merge(self) -> Optional[Geometry]: ...

	def render(self, size:Vec2d) -> str:
		xs = ''.join(check(self.merge()).render_items(size, Vec2d()))
		return f'{Geometry.checksum(xs):04x},{xs}'

	@abc.abstractmethod
	def render_item(self, size:Vec2d, posn:Vec2d) -> Iterator[str]: ...

	def render_items(self, size:Vec2d, posn:Vec2d) -> Iterator[str]:
		yield f'{size.x}x{size.y},{posn.x},{posn.y}'
		yield from self.render_item(size, posn)

	@classmethod
	def parse(cls, string:str) -> tuple[Vec2d,Geometry]:
		def parse_int(start:int) -> tuple[int,int]:
			assert string[start].isdigit(), f'expected digit at index {start}'
			end = start
			while end < len(string) and string[end].isdigit(): end += 1
			return end, int(string[start:end])
		def parse_char(start:int, chars:str) -> tuple[int,str]:
			assert string[start] in chars, f'expected [{chars}] at index {start}'
			return start + 1, string[start]
		def parse_geometry(start:int) -> tuple[int,Vec2d,Vec2d,Geometry]:
			index, width  = parse_int(start)
			index, height = parse_int(parse_char(index, 'x')[0])
			index, x      = parse_int(parse_char(index, ',')[0])
			index, y      = parse_int(parse_char(index, ',')[0])
			geometry: Geometry
			if string[index] in '[{':
				left_index = index
				direction = {'{': Direction.Horizontal, '[': Direction.Vertical}[string[index]]
				subgeometrys = []
				while index == left_index or string[index] == ',':
					index, size, posn, subgeometry = parse_geometry(index + 1)
					subgeometrys.append((size, posn, subgeometry))
				index, _ = parse_char(index,
					{Direction.Horizontal: '}', Direction.Vertical: ']'}[direction])
				assert len(subgeometrys) > 1
				if direction == Direction.Horizontal:
					assert len(set(size.y for size, _, _ in subgeometrys)) == 1
					assert len(set(posn.y for _, posn, _ in subgeometrys)) == 1
					assert len(set(posn.x for _, posn, _ in subgeometrys)) > 1
					assert width + 1 == sum(size.x for size, _, _ in subgeometrys) + len(subgeometrys)
					geometry = Split.horizontal(*((size.x, item) for size, _, item in subgeometrys))
				else:
					assert len(set(size.x for size, _, _ in subgeometrys)) == 1
					assert len(set(posn.x for _, posn, _ in subgeometrys)) == 1
					assert len(set(posn.y for _, posn, _ in subgeometrys)) > 1
					assert height + 1 == sum(size.y for size, _, _ in subgeometrys) + len(subgeometrys)
					geometry = Split.vertical(*((size.y, item) for size, _, item in subgeometrys))
			else:
				index, num = parse_int(parse_char(index, ',')[0])
				geometry = Single(f'%{num}')
			return index, Vec2d(width, height), Vec2d(x, y), geometry
		assert hex(cls.checksum(string[5:]))[2:] == string[:4]
		_, _ = parse_char(4, ',')
		index, size, posn, geometry = parse_geometry(5)
		assert posn.x == 0 and posn.y == 0
		assert index == len(string), f'expected EOF at index {index}'
		return size, geometry

@dataclasses.dataclass
class Single(Geometry):
	id: PaneID

	def iter(self) -> Iterator[PaneID]:
		yield self.id

	def mirror(self) -> Geometry:
		return self

	def merge(self) -> Optional[Geometry]:
		return self

	def render_item(self, size:Vec2d, posn:Vec2d) -> Iterator[str]:
		yield f',{self.id[1:]}'

@dataclasses.dataclass
class Split(Geometry):
	dir: Direction
	items: tuple[tuple[int,Geometry],...]

	def iter(self) -> Iterator[PaneID]:
		for size, item in self.items:
			yield from item.iter()

	def mirror(self) -> Geometry:
		return Split(self.dir.mirror(),
			tuple((size, item.mirror()) for size, item in self.items))

	def merge(self) -> Optional[Geometry]:
		items: list[tuple[int,Geometry]] = []
		for size, item in self.merge_items(self.dir):
			if size == 0: continue
			merged = item.merge()
			if merged is None: continue
			items.append((size, merged))
		if len(items) == 0: return None
		if len(items) == 1: return items[0][1]
		return Split(self.dir, tuple(items))

	def merge_items(self, direction:Direction) -> Iterator[tuple[int,Geometry]]:
		for size, item in self.items:
			if isinstance(item, Split) and item.dir == direction:
				yield from item.merge_items(direction)
			else:
				yield size, item

	def render_item(self, size:Vec2d, posn:Vec2d) -> Iterator[str]:
		yield {Direction.Horizontal: '{', Direction.Vertical: '['}[self.dir]
		comma, cursor = '', (posn.x if self.dir == Direction.Horizontal else posn.y)
		for (subsize, subgeometry) in self.items:
			yield comma; comma = ','
			if self.dir == Direction.Horizontal:
				args = Vec2d(subsize, size.y), Vec2d(cursor, posn.y)
			else:
				args = Vec2d(size.x, subsize), Vec2d(posn.x, cursor)
			yield from subgeometry.render_items(*args)
			cursor += subsize + 1
		expect = posn.x + size.x if self.dir == Direction.Horizontal else posn.y + size.y
		assert cursor - 1 == expect, 'bad size sum'
		yield {Direction.Horizontal: '}', Direction.Vertical: ']'}[self.dir]

	@dataclasses.dataclass
	class SplitHelper:
		dir: Direction

		def __call__(self, *subgeometrys:tuple[int,Geometry]) -> Split:
			assert len(subgeometrys)
			return Split(self.dir, subgeometrys)

		def even(self, size:int, *subgeometrys:Geometry) -> Split:
			assert len(subgeometrys)
			size -= len(subgeometrys) - 1
			div, mod = divmod(size, len(subgeometrys))
			return Split(self.dir, tuple((div + int(n < mod), subgeometry)
				for n, subgeometry in enumerate(subgeometrys)))

		def frac(self, size:int, lfrac:Fraction, left:Geometry, right:Geometry) -> Split:
			lsize = max(1, min(size - 2, round(lfrac * (size - 1))))
			return Split(self.dir, ((lsize, left), (size - lsize - 1, right)))

	horizontal: ClassVar[SplitHelper] = SplitHelper(Direction.Horizontal)
	vertical: ClassVar[SplitHelper] = SplitHelper(Direction.Vertical)

@dataclasses.dataclass
class WindowSet:
	focus: int
	stack: tuple[PaneID,...]
	hidden: tuple[PaneID,...]

	@dataclasses.dataclass
	class PanesHelper:
		windows: WindowSet

		@overload
		def __getitem__(self, ns:int) -> Single: ...
		@overload
		def __getitem__(self, ns:slice) -> tuple[Single,...]: ...
		def __getitem__(self, ns):
			if isinstance(ns, int):
				return Single(self.windows.stack[ns])
			return tuple(map(Single, self.windows.stack[ns]))

	@property
	def panes(self) -> PanesHelper:
		return WindowSet.PanesHelper(self)

	def unhide(self) -> WindowSet:
		return WindowSet(self.focus, self.stack + self.hidden, tuple())

class Layout(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry: ...

	def handle(self, session:Session, cmd:str, args:str) -> bool:
		return False

	def description(self) -> str:
		return type(self).__name__

	def __or__(self, other:Layout) -> Layout:
		if isinstance(self, Choice):
			if isinstance(other, Choice):
				return Choice((*self.layouts, *other.layouts), self.index)
			return Choice((*self.layouts, other), self.index)
		if isinstance(other, Choice):
			return Choice((self, *other.layouts), other.index + 1)
		return Choice((self, other))

@dataclasses.dataclass
class Choice(Layout):
	layouts: Sequence[Layout]
	index: int = 0

	def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry:
		return self.layouts[self.index].run(session, windows, size)

	def handle(self, session:Session, cmd:str, args:str) -> bool:
		if cmd == 'next_layout':
			if not args: args = '1'
			inc, wrap = splitline(args)
			self.index += int(inc)
			if wrap: self.index %= len(self.layouts)
			else: self.index = max(0, min(len(self.layouts) - 1, self.index))
			session.run()
			return True
		if cmd == 'jump_to_layout':
			if args.isdigit():
				self.index = int(args) % len(self.layouts)
			else:
				self.index = next(n for n, l in enumerate(self.layouts)
					if l.description() == args)
			session.run()
			return True
		return self.layouts[self.index].handle(session, cmd, args)

	def description(self) -> str:
		return self.layouts[self.index].description()

class Full(Layout):
	def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry:
		return windows.panes[windows.focus]

@dataclasses.dataclass
class Tall(Layout):
	nmaster: int = 1
	ratio: Fraction = Fraction(1, 2)
	increment: Fraction = Fraction(1, 30)

	def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry:
		windows = windows.unhide()
		nmaster = len(windows.stack) if self.nmaster == 0 else self.nmaster
		master = Split.vertical.even(size.y, *windows.panes[:nmaster])
		if len(windows.stack) <= self.nmaster: return master
		subs = Split.vertical.even(size.y, *windows.panes[self.nmaster:])
		return Split.horizontal.frac(size.x, self.ratio, master, subs)

	def handle(self, session:Session, cmd:str, args:str) -> bool:
		if cmd == 'inc_master':
			self.nmaster = max(0, self.nmaster + int(args))
			session.run()
			return True
		if cmd == 'resize':
			self.ratio = max(Fraction(0), min(Fraction(1),
				self.ratio + int(args) * self.increment))
			session.run()
			return True
		return False

@dataclasses.dataclass
class Mirror(Layout):
	layout: Layout

	def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry:
		return self.layout.run(session, windows, size.mirror()).mirror()

	def handle(self, session:Session, cmd:str, args:str) -> bool:
		return self.layout.handle(session, cmd, args)

	def description(self) -> str:
		return f'Mirror({self.layout.description()})'

@dataclasses.dataclass
class Accordion(Layout):
	collapsed: int = 2
	max_frac: Fraction = Fraction(1, 2)

	def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry:
		windows = windows.unhide()
		bits: list[tuple[int,Geometry]] = []
		for pane in windows.panes[:windows.focus]:
			bits.append((self.collapsed, pane))
		middle = size.y - (self.collapsed + 1) * (len(windows.stack) - 1)
		bits.append((middle, windows.panes[windows.focus]))
		for pane in windows.panes[windows.focus+1:]:
			bits.append((self.collapsed, pane))
		return Split.vertical(*bits)

	def handle(self, session:Session, cmd:str, args:str) -> bool:
		if cmd == 'resize':
			self.collapsed = max(1, self.collapsed + int(args))
			session.run()
			return True
		return False

# class TwoPane(Layout):
	# ratio: Fraction = Fraction(1, 2)
	# increment: Fraction = Fraction(1, 30)

	# def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry:
		# pass

	# def handle(self, session:Session, cmd:str, args:str) -> bool:
		# if cmd == 'resize':
			# self.ratio = max(Fraction(0), min(Fraction(1),
				# self.ratio + int(args) * self.increment))
			# session.run()
			# return True
		# return False

# class Combine(Layout):
	# comb: Layout
	# left: Layout
	# right: Layout

	# def run(self, session:Session, windows:WindowSet, size:Vec2d) -> Geometry:
		# pass

	# def handle(self, session:Session, cmd:str, args:str) -> bool:
		# pass

	# def description(self) -> str:
		# return f'Combine({self.comb.description()}, ' \
			# + f'{self.left.description()}, {self.right.description()})'

@dataclasses.dataclass
class PauseHook:
	server: Server
	hooks: Iterable[str]
	paused: bool = False

	def __call__(self) -> PauseHook:
		if not self.paused:
			self.paused = True
			for hook in self.hooks:
				self.server.unhook(hook)
		return self

	def __enter__(self) -> PauseHook:
		return self

	def __exit__(self, exc_type, exc_value, traceback) -> None:
		if not self.paused: return
		for hook in self.hooks:
			self.server.hook(hook)

@dataclasses.dataclass
class Session:
	tmux: libtmux.Session
	server: Server
	layouts: dict[WindowID,Layout] = dataclasses.field(default_factory=dict)

	def layout(self, window:Optional[Union[WindowID,libtmux.Window]]=None) -> Layout:
		if window is None:
			window = check(self.tmux.active_window)
		if isinstance(window, libtmux.Window):
			window = check(window.id)
		if window not in self.layouts:
			self.layouts[window] = self.server.manager.layout(check(self.tmux.session_name),
				int(check(check(self.tmux.windows.get(window_id=window)).window_index)))
		return self.layouts[window]

	def reorder(self,
		windows:WindowSet, target:list[PaneID],
		window:Optional[libtmux.Window]=None,
		hindex:Optional[int]=None
	) -> bool:
		logging.info(('reorder', target, windows))
		active = windows.stack[windows.focus]
		source = list(windows.stack)
		if window is None: window = check(self.tmux.active_window)
		if hindex is None: hindex = int(check(window.window_index)) \
			+ self.server.manager.hide_offset
		cmds: list[tuple[str,...]] = []
		hidden = list(windows.hidden)
		for pane in set(target) - set(source):
			logging.info(('move-in', pane))
			cmds.append(('select-layout', '-t', source[-1], 'tiled'))
			cmds.append(('join-pane', '-d', '-h', '-s', pane, '-t', source[-1]))
			source.append(pane) ; hidden.remove(pane)
		for pane in set(source) - set(target):
			logging.info(('move-out', pane))
			if hidden:
				cmds.append(('select-layout', '-t', hidden[-1], 'tiled'))
				cmds.append(('join-pane', '-d', '-h', '-s', pane, '-t', hidden[-1]))
			else: cmds.append(('break-pane', '-d', '-s', pane, '-t', str(hindex)))
			hidden.append(pane) ; source.remove(pane)
		assert set(source) == set(target)
		if source.index(active) != target.index(active):
			logging.info(f'reselect {active}')
			cmds.append(('select-pane', '-t', str(active)))
		for n, pane in enumerate(target):
			if pane == source[n]: continue
			m = source.index(pane, n)
			logging.info(('swap', source[m], source[n]))
			cmds.append(('swap-pane', '-d', '-s', source[m], '-t', source[n]))
			source[m], source[n] = source[n], source[m]
		if cmds:
			with PauseHook(self.server, ('window-pane-changed',), paused=True)():
				logging.info(cmds)
				self.tmux.cmd(*(bit for cmd in cmds for bit in (*cmd, ';')))
		return bool(cmds)

	def windowset(self, window:Optional[libtmux.Window]=None) -> WindowSet:
		if window is None: window = check(self.tmux.active_window)
		panes = tuple(check(pane.id) for pane in window.panes)
		active = check(check(window.active_pane).id)
		hindex = int(check(window.window_index)) + self.server.manager.hide_offset
		try: hwindow = check(self.tmux.windows.get(window_index=str(hindex)))
		except: hidden: tuple[PaneID,...] = tuple()
		else: hidden = tuple(check(pane.id) for pane in hwindow.panes)
		return WindowSet(panes.index(active), panes, hidden)

	def run(self) -> None:
		window = check(self.tmux.active_window)
		layout = self.layout(check(window.id))
		logging.info(layout.description())
		size = Vec2d(int(check(window.window_width)), int(check(window.window_height)))
		windows = self.windowset(window)
		logging.info(windows)
		geometry = layout.run(self, windows, size)
		logging.info(geometry)
		self.reorder(windows, list(geometry.iter()), window)
		window.select_layout(geometry.render(size))

@dataclasses.dataclass
class Server:
	pid: str
	tmux: libtmux.Server
	manager: Manager
	sessions: dict[SessionID,Session] = dataclasses.field(default_factory=dict)

	def hook(self, name:str) -> None:
		# return
		cmd = f'run-shell "printf \\"%s\\n\\" hook-{name} \\$TMUX | nc -NU {self.manager.socket}"'
		logging.info(('hook', name, cmd))
		self.tmux.cmd('set-hook', '-g', name, cmd)
	def unhook(self, name:str) -> None:
		# return
		logging.info(('unhook', name))
		self.tmux.cmd('set-hook', '-u', '-g', name)

	def __getitem__(self, session:Union[SessionID,libtmux.Session]) -> Session:
		if isinstance(session, libtmux.Session):
			session = check(session.id)
		if session not in self.sessions:
			self.sessions[session] = Session(
				check(self.tmux.sessions.get(session_id=session)), self)
		return self.sessions[session]

@dataclasses.dataclass
class Manager:
	socket: str
	layout: Callable[[SessionName,WindowIndex],Layout]
	hide_offset: int = 10
	servers: dict[ServerPath,Server] = dataclasses.field(default_factory=dict)

	def load(self, servers):
		for pid, socket, layouts in servers:
			try: tmux = libtmux.Server(socket_path=socket)
			except: pass
			else: self.servers[socket] = Server(pid, tmux, self, layouts)
		return self

	def dump(self):
		return [(path, server.pid, server.sessions)
			for path, server in self.servers.items()]

	def __getitem__(self, item:tuple[str,str]) -> Server:
		socket, pid = item
		if socket in self.servers:
			if self.servers[socket].pid != pid:
				del self.servers[socket]
		if socket not in self.servers:
			self.servers[socket] = Server(pid,
				libtmux.Server(socket_path=socket), self)
		return self.servers[socket]

	def handle(self, message:str) -> None:
		if message[-1] == '\n': message = message[:-1]
		logging.info(message)
		cmd, message = splitline(message)
		handlers[cmd].handle(self, message)

handlers: dict[str,Handler] = {}

@dataclasses.dataclass
class Handler(Generic[T], metaclass=abc.ABCMeta):
	func: T
	@classmethod
	def add(cls, name:str):
		def inner(func:T) -> T:
			global handlers
			assert name not in handlers
			handlers[name] = cls(func)
			return func
		return inner
	@abc.abstractmethod
	def handle(self, manager:Manager, message:str) -> None: ...

class HandlerMessage(Handler[Callable[[Manager,str],None]]):
	def handle(self, manager:Manager, message:str) -> None:
		self.func(manager, message)
message_handler = HandlerMessage.add

class HandlerServer(Handler[Callable[[Server,str],None]]):
	def handle(self, manager:Manager, message:str) -> None:
		tmux, message = splitline(message)
		socket, pid, sessionidx = tmux.rsplit(',', 2)
		server = manager[socket, pid]
		self.func(server, message)
server_handler = HandlerServer.add

class HandlerSession(Handler[Callable[[Session,str],None]]):
	def handle(self, manager:Manager, message:str) -> None:
		tmux, message = splitline(message)
		socket, pid, sessionidx = tmux.rsplit(',', 2)
		server = manager[socket, pid]
		session = server[f'${sessionidx}']
		self.func(session, message)
session_handler = HandlerSession.add

@session_handler('layout')
def handler_layout(session:Session, message:str) -> None:
	session.run()

@session_handler('message')
def handler_message(session:Session, message:str) -> None:
	cmd, args = splitline(message)
	session.layout().handle(session, cmd, args)

@server_handler('start')
def handler_start(server:Server, message:str) -> None:
	# after-bind-key           after-capture-pane     after-copy-mode
	# after-display-message    after-display-panes    after-kill-pane
	# after-list-buffers       after-list-clients     after-list-keys
	# after-list-panes         after-list-sessions    after-list-windows
	# after-load-buffer        after-lock-server      after-new-session
	# after-new-window         after-paste-buffer     after-pipe-pane
	# after-queue              after-refresh-client   after-rename-session
	# after-rename-window      after-resize-pane      after-resize-window
	# after-save-buffer        after-select-layout    after-select-pane
	# after-select-window      after-send-keys        after-set-buffer
	# after-set-environment    after-set-hook         after-set-option
	# after-show-environment   after-show-messages    after-show-options
	# after-split-window       after-unbind-key       alert-activity
	# alert-bell               alert-silence          client-active
	# client-attached          client-detached        client-focus-in
	# client-focus-out         client-resized         client-session-changed
	# pane-died                pane-exited            pane-focus-in
	# pane-focus-out           pane-mode-changed      pane-set-clipboard
	# session-closed           session-created        session-renamed
	# session-window-changed   window-linked          window-pane-changed
	# window-renamed           window-resized         window-unlinked
	server.hook('after-split-window')
	server.hook('session-created')
	server.hook('session-closed')
	server.hook('session-window-changed')
	server.hook('window-pane-changed')
	server.hook('window-resized')

@session_handler('hook-session-created')
def handler_hook_session_created(session:Session, message:str) -> None:
	pass

@server_handler('hook-session-closed')
def handler_hook_session_closed(server:Server, message:str) -> None:
	sessions = set(s.session_id for s in server.tmux.sessions)
	for s in list(server.sessions):
		if s not in sessions:
			del server.sessions[s]

@session_handler('hook-after-split-window')
@session_handler('hook-session-window-changed')
@session_handler('hook-window-pane-changed')
@session_handler('hook-window-resized')
def handler_hook_run_layout(session:Session, message:str) -> None:
	session.run()

@message_handler('echo')
def handler_echo(manager:Manager, message:str) -> None:
	pass

@message_handler('exit')
def handler_exit(manager:Manager, message:str) -> None:
	sys.exit(0)

def main():
	import socketserver
	import os
	import atexit
	import argparse
	import importlib
	import tempfile

	parser = argparse.ArgumentParser(prog='tmux-layout')
	parser.add_argument('--log-level', default='INFO')
	parser.add_argument('--log-file',
		default=sys.stdout, type=argparse.FileType('w+'))
	parser.add_argument('--replace', action='store_false')
	parser.add_argument('--socket',
		default=os.path.join(tempfile.gettempdir(),
			f'tmux-layout-{os.getuid()}'))
	parser.add_argument('--config',
		default=os.path.join(os.getenv("XDG_CONFIG_HOME",
			os.path.expanduser("~/.config")), 'tmux/layout.py'))
	args = parser.parse_args()

	logging.basicConfig(datefmt='%H:%M:%S',
		level=getattr(logging, args.log_level),
		stream=args.log_file,
		format='\x1b[30;105m %(asctime)s.%(msecs)d \x1b[95;104m\ue0b0'
			'\x1b[30;104m %(levelname)s \x1b[94;107m\ue0b0'
			'\x1b[30;107m %(filename)s:%(lineno)d:%(funcName)s \x1b[97;49m\ue0b0'
			'\x1b[0m\n%(message)s')
	logging.info(f'args: {args}')

	if os.path.exists(args.socket):
		logging.warning(f'layout server found at {args.socket}')
		if not args.replace: return
		os.system('echo exit | nc -NU {args.socket}')

	assert os.path.exists(args.config)
	sys.path.insert(0, os.path.dirname(args.config))
	sys.path.insert(0, os.path.dirname(__file__))
	assert 'layout' not in sys.modules
	config = importlib.import_module('layout')

	manager = Manager(args.socket, config.layout)
	logging.info(f'manager: {manager}')

	class SocketHandler(socketserver.StreamRequestHandler):
		def handle(self):
			try: manager.handle(self.rfile.read().decode('utf-8'))
			except Exception as exc: logging.error('exception', exc_info=exc)

	if os.path.exists(args.socket): os.unlink(args.socket)
	with socketserver.UnixStreamServer(args.socket, SocketHandler) as server:
		atexit.register(os.unlink, args.socket)
		logging.info(f'started server at {args.socket}')
		server.serve_forever()

if __name__ == '__main__':
	main()
