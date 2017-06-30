from pkg.pinyin_tone.pinyin import EngTonedMarker

marker = EngTonedMarker()
ret = marker.mark("你是谁hello")
print(ret)
assert(ret == ['ni3', 'shi4', 'shui2', 'h', 'e', 'l', 'l', 'o']) 
