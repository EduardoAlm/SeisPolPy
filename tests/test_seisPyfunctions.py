from SeisPolPy import seisPolPyfunctions
import mat4py

def test_haversine():
    assert seisPolPyfunctions.flinn(52.370216, 4.895168, 52.520008,13.404954) == 945793.4375088713
