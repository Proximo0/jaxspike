class_name FeelAudio
extends Node
## Audio hooks of the world view: a few pooled AudioStreamPlayers that play the AudioCues of every new server tick
## (walk, attack, hit, level-up, pickup, warp) from `res://assets/audio/<cue>.wav`. Instance it under GameView and
## call `bind(session)`; `play(cue)` is public for other systems (UI clicks later). A missing file is skipped.

const AUDIO_DIR: String = "res://assets/audio/"
const VOICES: int = 4
const VOLUME_DB: Dictionary = {"walk": -14.0, "attack": -6.0, "hit": -6.0, "level_up": -3.0, "pickup": -6.0, "warp": -6.0}

var session: NetSession = null
var cues: AudioCues = AudioCues.new()
var muted: bool = false
var played: Array[String] = []  ## Every cue played since bind (for sims and the overlay); capped at 64.
var streams: Dictionary = {}  ## cue -> AudioStream
var _voices: Array[AudioStreamPlayer] = []
var _next: int = 0
var _tick: int = -1


func _ready() -> void:
	for cue: String in AudioCues.ALL:
		var path: String = AUDIO_DIR + cue + ".wav"
		if ResourceLoader.exists(path):
			streams[cue] = load(path)
	for i: int in range(VOICES):
		var p := AudioStreamPlayer.new()
		add_child(p)
		_voices.append(p)


func bind(s: NetSession) -> void:
	session = s
	cues.reset()
	played.clear()
	_tick = -1


func _process(_delta: float) -> void:
	if session == null or session.store.tick == _tick:
		return
	_tick = session.store.tick
	for cue: String in cues.observe(session.store.you, session.store.get_ent(session.slot)):
		play(cue)


func play(cue: String) -> void:
	played.append(cue)
	if played.size() > 64:
		played.pop_front()
	if muted or not streams.has(cue) or _voices.is_empty():
		return
	var p: AudioStreamPlayer = _voices[_next]
	_next = (_next + 1) % _voices.size()
	p.stream = streams[cue]
	p.volume_db = float(VOLUME_DB.get(cue, -6.0))
	p.play()
