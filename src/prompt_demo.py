import sys
import functools
import asyncio as aio
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Prompt:
    def __init__(self, loop=None):
        self.loop = loop or aio.get_event_loop()
        self.q = aio.Queue()
        self.loop.add_reader(sys.stdin, self.got_input)

    def got_input(self):
        aio.ensure_future(self.q.put(sys.stdin.readline()), loop=self.loop)

    async def __call__(self, msg, end='\n', flush=False):
        print(msg, end=end, flush=flush)
        return (await self.q.get()).rstrip('\n')

prompt = Prompt()
raw_input = functools.partial(prompt, end='', flush=True)

async def main():
    # wait for user to press enter
    await prompt("press enter to continue")

    # simulate raw_input
    print(await raw_input('enter something:'))

loop = aio.get_event_loop()
loop.run_until_complete(main())
loop.close()