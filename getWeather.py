# Retrieves weather conditions from weather stations


import urllib.request

contents = urllib.request.urlopen('http://weather.sun.ac.za/api/getlivedata.php?time&winddir&windsec&wind').read()


print(contents)
